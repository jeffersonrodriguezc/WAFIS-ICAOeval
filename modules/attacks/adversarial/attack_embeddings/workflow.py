from fileinput import filename
import os
import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from PIL import Image

from dataset import WatermarkedDataset, FaceAttackedDataset
from options.options import InjectionOptions
from utils import l2_norm, alignment, tensor2img, pgd_step, l2_project, pgd_step_linf, linf_project, pgd_step_linf_masked
from utils import compute_sobel_edges_mask, generate_face_box_mask
from network.AAD import AADGenerator, FusionModule, get_spatial_weights_gauss
from network.MAE import MLAttrEncoder
from criteria.loss_functions import RecLoss, AdvLoss, FreqLoss
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torch.optim.lr_scheduler import ExponentialLR
import torchvision.utils as vutils

# temporal - to load the stegaformer decoder, we need to import the functions to build the model and load the weights
import sys
sys.path.insert(0, '/app/watermarking/stegaformer')

from watermarking.StegFormer.utils import get_message_accuracy as get_message_accuracy_StegFormer
from watermarking.StegFormer.utils import load_weights_decoder as load_weights_StegFormer
from watermarking.StegFormer.model import build_models as build_stegformer_models
from watermarking.stegaformer.stegaformer import build_models as build_stegaformer_models
from watermarking.stegaformer.utils import load_weights_decoder as load_weights_StegFormer 
from watermarking.stegaformer.utils import get_message_accuracy as get_message_accuracy_StegaFormer
from recognition.arcface.recognizer import ArcFaceRecognizer, compare_crops_interactive as compare_crops_interactive_arcface
from recognition.facenet.recognizer import FaceNetRecognizer, compare_crops_interactive as compare_crops_interactive_facenet

class AttackEmbeddings:
    def __init__(self, opts, wm_args):
        torch.cuda.empty_cache()

        self.opts = opts
        self.wm_args = wm_args
        self.device = torch.device(opts.device)
        self.global_step = 0
        self.start_epoch = 0
        self.inner_step_count = 0
        self.val_step_count = 0
        self.best_global_loss = float('inf')  
        self._set_seeds()
      
        # Directories to save results, checkpoints and logs
        if self.opts.baseline:
            self.folder_struct = "baseline"
        else:
            self.folder_struct = "pipeline"

        self.output_dir = opts.output_dir
        self.output_to_save = opts.output_to_save

        # create the folders if they don't exist
        if not os.path.exists(os.path.join(self.output_to_save, self.folder_struct)):
            self.id_number_exp = "1"
            self.imgout_dir = os.path.join(self.output_to_save, self.folder_struct, self.id_number_exp, 'attacked_samples')
            self.delta_dir = os.path.join(self.output_to_save, self.folder_struct, self.id_number_exp, 'deltas_samples')
            #self.records_dir = os.path.join(self.output_to_save, self.folder_struct, self.id_number_exp, 'attack_records')
            self.logs_dir = os.path.join(self.output_to_save, self.folder_struct, self.id_number_exp, 'logs')
            self.ckpt_dir = os.path.join(self.output_to_save, self.folder_struct, self.id_number_exp, 'checkpoints')
            # create folders
            os.makedirs(self.imgout_dir, exist_ok=True)
            os.makedirs(self.delta_dir, exist_ok=True)
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.makedirs(self.logs_dir, exist_ok=True)
            #os.makedirs(self.records_dir, exist_ok=True)
            os.makedirs(os.path.join(self.imgout_dir, 'train'), exist_ok=True)
            os.makedirs(os.path.join(self.imgout_dir, 'test'), exist_ok=True)
            if self.opts.baseline == False:
                os.makedirs(os.path.join(self.imgout_dir, 'val'), exist_ok=True)   
        else:
            # Read if there are created folders to create an id for the current experiment
            last_id_exp = len(os.listdir(os.path.join(self.output_to_save, self.folder_struct)))
            if self.opts.restore_training == False and self.opts.use_fusion_module == True and self.opts.baseline == False:
                self.id_number_exp = str(int(last_id_exp) + 1)
                self.imgout_dir = os.path.join(self.output_to_save, self.folder_struct, self.id_number_exp, 'attacked_samples')
                self.delta_dir = os.path.join(self.output_to_save, self.folder_struct, self.id_number_exp, 'deltas_samples')
                #self.records_dir = os.path.join(self.output_to_save, self.folder_struct, self.id_number_exp, 'attack_records')
                self.logs_dir = os.path.join(self.output_to_save, self.folder_struct, self.id_number_exp, 'logs')
                self.ckpt_dir = os.path.join(self.output_to_save, self.folder_struct, self.id_number_exp, 'checkpoints')
                # create folders
                os.makedirs(self.imgout_dir, exist_ok=True)
                os.makedirs(self.delta_dir, exist_ok=True)
                os.makedirs(self.ckpt_dir, exist_ok=True)
                os.makedirs(self.logs_dir, exist_ok=True)
                #os.makedirs(self.records_dir, exist_ok=True)
                os.makedirs(os.path.join(self.imgout_dir, 'train'), exist_ok=True)
                os.makedirs(os.path.join(self.imgout_dir, 'test'), exist_ok=True)
                os.makedirs(os.path.join(self.imgout_dir, 'val'), exist_ok=True)
            elif self.opts.restore_training == True and self.opts.baseline == False and self.opts.use_fusion_module == True:
                self.id_number_exp = str(last_id_exp)
                self.imgout_dir = os.path.join(self.output_to_save, self.folder_struct, self.id_number_exp, 'attacked_samples')
                self.delta_dir = os.path.join(self.output_to_save, self.folder_struct, self.id_number_exp, 'deltas_samples')
                #self.records_dir = os.path.join(self.output_to_save, self.folder_struct, self.id_number_exp, 'attack_records')
                self.logs_dir = os.path.join(self.output_to_save, self.folder_struct, self.id_number_exp, 'logs')
                self.ckpt_dir = os.path.join(self.output_to_save, self.folder_struct, self.id_number_exp, 'checkpoints')            
            elif self.opts.baseline == True and self.opts.only_face_recognition_evaluation == False:
                self.id_number_exp = str(int(last_id_exp) + 1)
                self.imgout_dir = os.path.join(self.output_to_save, self.folder_struct, self.id_number_exp, 'attacked_samples')
                self.delta_dir = os.path.join(self.output_to_save, self.folder_struct, self.id_number_exp, 'deltas_samples')
                self.logs_dir = os.path.join(self.output_to_save, self.folder_struct, self.id_number_exp, 'logs')
                # create folders
                os.makedirs(self.imgout_dir, exist_ok=True)
                os.makedirs(self.delta_dir, exist_ok=True)
                os.makedirs(self.logs_dir, exist_ok=True)
                #os.makedirs(self.records_dir, exist_ok=True)
                os.makedirs(os.path.join(self.imgout_dir, 'train'), exist_ok=True)
                os.makedirs(os.path.join(self.imgout_dir, 'test'), exist_ok=True)
            elif self.opts.baseline == True and self.opts.only_face_recognition_evaluation == True:
                self.id_number_exp = str(self.opts.id_number_exp)
                self.imgout_dir = os.path.join(self.output_to_save, self.folder_struct, self.id_number_exp, 'attacked_samples')
                
        # save all the parameters of the experiment in a json file for later reference
        if self.opts.restore_training == False or self.opts.only_face_recognition_evaluation == True:
            self.save_params_to_json() 
        else:
            self.load_params_from_json()    

        print("[*] Initializing Networks...") 
        # Face Recognition is always needed!

        print(f"[*] 1a. FaceNet Model ({opts.facenet_mode}) from {opts.facenet_dir}")
        print(f"[*] 1b. FaceNet Model for test ({opts.facenet_mode_test})")
        self.facenet = self._load_facenet(self.opts.facenet_mode, self.opts.facenet_dir)
        self.facenet_test = self._load_facenet(self.opts.facenet_mode_test, self.opts.facenet_dir_test)

        # Watermark Decoder (Black Box) is always needed to compute extraction metrics
        print(f"[*] 2. Watermark Decoder ({opts.wm_algorithm})") 
        self.wm_decoder = self._load_wm_decoder(self.opts, self.wm_args).to(self.device).eval()      
        
        # If the baseline is False, we have to initialize all the networks
        # to use in the complete pipeline. 
        if self.opts.baseline == False:
            # initialize the base modules for the complete pipeline
            print(f"[*] 3. Attribute Encoder from {opts.attencoder_dir}")
            self.attencoder = MLAttrEncoder().to(self.device).eval()
            self.attencoder.load_state_dict(torch.load(opts.attencoder_dir, map_location=self.device))

            print(f"[*] 4. ADD Network from {opts.aadblocks_dir}")
            self.aadblocks = AADGenerator(c_id=512).to(self.device).eval()
            self.aadblocks.load_state_dict(torch.load(opts.aadblocks_dir, map_location=self.device))
            
            if self.opts.use_fusion_module:
                print(f"[*] 5. Fusion Network from scratch")
                # if we want to train the fusion module.
                self.fusion_net = FusionModule().to(self.device).train()  
                self.optimizer_fusion = torch.optim.Adam(
                    self.fusion_net.parameters(), 
                    lr=self.opts.lr_fusion
                )
                self.scheduler = ExponentialLR(self.optimizer_fusion, gamma=0.9)
                
                # Load checkpoint if exists for fusion module
                print("[*] Checking for existing checkpoints for the fusion module...")
                self._load_checkpoint(best=False)  # Load last checkpoint if exists, to resume training of the fusion module

                # Freeze the base modules (AAD and Attribute Encoder)
                # Here the only network that we could fine tune is the AAD network!
                for m in [self.aadblocks, self.attencoder]:
                    for p in m.parameters(): p.requires_grad = False

        # Freeze the FaceNet and Watermark Decoder
        for m in [self.facenet.model, self.facenet_test.model, self.wm_decoder]:
            for p in m.parameters(): p.requires_grad = False

        print("[*] 7. Loss functions and metrics...")
        # loss functions and metrics
        self.rec_loss = RecLoss(opts.rec_weight, opts.recloss_mode, self.device, opts.mse_weight, opts.lpips_weight)
        self.adv_loss = AdvLoss(opts.adv_weight, self.device, mode='evasion')
        self.freq_loss = FreqLoss(opts.freq_weight, device = self.device, loss_mode=opts.loss_mode, band="all", gamma=1.0)

        if self.opts.wm_algorithm.lower() == 'stegaformer':
            self.data_range = 255.0
        elif self.opts.wm_algorithm.lower() == 'facesings':
            self.data_range = 1.0
        elif self.opts.wm_algorithm.lower() == 'stegformer':
            self.data_range = 1.0

        self.cal_psnr = PeakSignalNoiseRatio(data_range=self.data_range).to(self.device)
        self.cal_ssim = StructuralSimilarityIndexMeasure(data_range=self.data_range).to(self.device)

        if self.opts.only_face_recognition_evaluation == False:
            print("[*] 8. Tensorboard Writer...")
            self.writer = SummaryWriter(log_dir=self.logs_dir,
                                    purge_step=self.global_step if self.opts.restore_training else None)
            self.per_step_loss_buffer = []
            self.per_step_delta_buffer = []
            self.per_step_lrec_buffer = []
            self.per_step_ladv_buffer = []
            self.per_step_lfreq_buffer = []

        # create datasets and dataloaders
        print("[*] Preparing Datasets and Dataloaders...")
        self.create_sets() # for the attack only

    def _save_checkpoint(self, epoch, loss, is_best=False):
        """
        This function saves the current state of the training, including model weights, optimizer state, scheduler state, and training progress.
        """
        state = {
                    'epoch': epoch,
                    'global_step': self.global_step,
                    'inner_step_count': self.inner_step_count,
                    'val_step_count': self.val_step_count,
                    'model_state': self.fusion_net.state_dict(),
                    'optimizer_state': self.optimizer_fusion.state_dict(),
                    'scheduler_state': self.scheduler.state_dict(),
                    'best_loss': self.best_global_loss,
                    'current_loss': loss  
                }
        
        last_path = os.path.join(self.ckpt_dir, 'last_checkpoint.pth')
        torch.save(state, last_path)
        
        if is_best:
            best_path = os.path.join(self.ckpt_dir, 'best_checkpoint.pth')
            torch.save(state, best_path)
            #print(f"[*] New Best Model Saved! Loss: {loss:.4f}")

    def _load_checkpoint(self, best=False):
        """
        This function loads the last checkpoint if it exists, allowing the training to resume from where it left off.
        """
        filename = 'best_checkpoint.pth' if best else 'last_checkpoint.pth'
        load_path = os.path.join(self.ckpt_dir, filename)
        if os.path.exists(load_path):
                tag_msg = "BEST" if best else "LAST"
                print(f"[*] Loading {tag_msg} checkpoint from {load_path}")
                checkpoint = torch.load(load_path, map_location=self.device)
                
                # Load model and optimizer state
                self.fusion_net.load_state_dict(checkpoint['model_state'])
                self.optimizer_fusion.load_state_dict(checkpoint['optimizer_state'])

                if 'scheduler_state' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state'])

                # Restore training progress
                self.start_epoch = checkpoint['epoch']
                self.global_step = checkpoint['global_step']
                self.inner_step_count = checkpoint.get('inner_step_count', 0)
                self.val_step_count = checkpoint.get('val_step_count', 0)
                self.best_global_loss = checkpoint.get('best_loss', float('inf'))
                
                print(f"[*] Resumed at Epoch {self.start_epoch}, Best Loss {self.best_global_loss:.4f}")
        else:
            print(f"[*] No checkpoint found at {load_path}. Skipping load.")

    def _set_seeds(self):
        torch.manual_seed(self.opts.seed)
        np.random.seed(self.opts.seed)
        torch.cuda.manual_seed_all(self.opts.seed)

    def _load_facenet(self, facenet_mode=None, facenet_dir=None):
        # Simplificación de carga basada en tu script original
        if facenet_mode == 'arcface':
            #net = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
            #net.load_state_dict(torch.load(facenet_dir, map_location=self.device))
            net = ArcFaceRecognizer(weight_path=facenet_dir, device=self.device, network='r100', use_mtcnn=True)

        elif facenet_mode == 'facenet':
            #net = Backbone_facenet(pretrained="vggface2").to(self.device)
            net = FaceNetRecognizer(device=self.device, use_mtcnn=True) #pretrained='vggface2'

        return net
    
    def _load_wm_decoder(self, args, wm_args):
        if args.wm_algorithm.lower() == 'facesings':
            # Load the watermark decoder
            pass

        elif args.wm_algorithm.lower() == 'stegformer':
            decoder = build_stegformer_models(wm_args, build='decoder')

            if os.path.exists(args.wm_model_path):
                print(f"[*] Loading WM Decoder from {args.wm_model_path}")
                load_weights_StegFormer(decoder, args.wm_model_path, tag=args.wm_tag)
            else:
                raise FileNotFoundError(f"[*] WM Decoder weights not found at {args.wm_model_path}")

        elif args.wm_algorithm.lower() == 'stegaformer':
            decoder = build_stegaformer_models(wm_args, build='decoder')

            if os.path.exists(args.wm_model_path):
                print(f"[*] Loading WM Decoder from {args.wm_model_path}")
                load_weights_StegFormer(decoder, args.wm_model_path, tag=args.wm_tag)
            else:
                raise FileNotFoundError(f"[*] WM Decoder weights not found at {args.wm_model_path}")
        return decoder

    def attack_batch_pipeline(self, img_org, tag, update_weights=False):
        """
        Pipeline computation. Attack directly in the embedding space with PGD, then
        using the AAD network to generate the adversarial image and optionally the fusion module to preserve the watermark and visual quality of the image.
        """
        # Set the status for the fusion module (training or evaluation) 
        if self.opts.use_fusion_module and update_weights and tag == 'train':
            self.fusion_net.train()
        else:
            self.fusion_net.eval()

        # Weight mask for the regularization of the fusion module (if we want to use it)
        if self.opts.use_weight_mask == True:
            #kernel = get_gaussian_kernel(kernel_size=3, sigma=0.5).to(self.device)
            # spatial weights to regularize the mask of the fusion module, giving more importance 
            # to the central region of the face
            self.spatial_weights = get_spatial_weights_gauss(self.opts.wm_image_size, 
                                                   self.opts.wm_image_size, 
                                                   self.device, sigma_scale=0.5, center_val=0.2)

        # 1. Get the original embedding as reference (zid) and the attribute arrays (zatt) before the attack
        with torch.no_grad():
            img_org_aligned = alignment(img_org) # to 112x112
            img_org_for_net = (img_org_aligned - 0.5) / 0.5 # ARCFace normalization
            zid = l2_norm(self.facenet(img_org_for_net)).detach()
            zatt = self.attencoder(img_org) # Attribute arrays by layers
            
        # 2. Initialize delta (perturbation in the embedding space)
        #delta = torch.zeros_like(zid, requires_grad=True)
        delta = torch.zeros_like(zid).to(self.device)
        delta.data.uniform_(-self.opts.epsilon, self.opts.epsilon)
        delta.requires_grad = True 

        best_attack = None
        best_loss = float('inf')
        # Loop PGD
        for i in range(self.opts.pgd_steps):
            # 3. Generate the adversarial image from the perturbed embedding
            x_adv = self.aadblocks(inputs=(zatt, zid + delta)) # using the AAD network

            # 4. if we are using the fusion module
            if self.opts.use_fusion_module:
                # The network learns how to fuse the adversarial image with the original one,
                #  to preserve the watermark and the visual quality
                if self.opts.use_weight_mask == True:
                    # if we want to use the weight mask, we pass it to the fusion module to 
                    # force it to learn a mask that gives more importance to the central region of the face, 
                    # rewarding to preserve the watermark inside the face.
                    x_final, mask = self.fusion_net(img_org, x_adv, self.spatial_weights)
                else:
                    x_final, mask = self.fusion_net(img_org, x_adv)
                
            # 5. Extract the embedding of the adversarial image to calculate the losses and metrics    
            if self.opts.use_fusion_module == True:
                # Should resize the adversarial image
                x_adv_aligned = alignment(x_final)
            else:
                x_adv_aligned = alignment(x_adv)

            # 6. Continue with the same normalization for ARCface
            x_adv_for_net = (x_adv_aligned - 0.5) / 0.5
            # And normaliation to facilitates the project step of the PGD in the embedding space
            zadv = l2_norm(self.facenet(x_adv_for_net))

            # 7. Compute losses and metrics
            # Term 2 of the loss: we want to preserve the watermark and the visual quality of the image, 
            # so we calculate the reconstruction loss with the original image. Similarity maximization.
            if self.opts.use_fusion_module:
                lrec = self.rec_loss(x_final, img_org) 
            else:
                lrec = self.rec_loss(x_adv, img_org)

            # Term 1: we want to minimize the similarity between the adversarial embedding and the original one (maximize the distance)
            ladv = self.adv_loss(zadv, zid) 
            
            # Term 3: Regularization of the mask of the fusion module, 
            # My intuition behind this is if we could force to learn more zero values (black mask part related to original image),
            # I could recovery more from the original watermark .. we want white zone related to the new adversarial identity
            # be the most small possible region, less regularization --> more black zone, more regularization --> more white zone.
            if self.opts.use_fusion_module and self.opts.use_weight_mask:
                loss_mask_reg = torch.mean(mask)
                # compute the total loss
                loss = ladv + lrec + (self.opts.mask_reg * loss_mask_reg)
            else:
                loss = ladv + lrec
            
            # 8. Backpropagation and PGD step
            if delta.grad is not None: delta.grad.zero_() # To avoid acumulation of gradients in the delta variable

            if self.opts.use_fusion_module and update_weights:
                self.optimizer_fusion.zero_grad() # To avoid acumulation of gradients in the fusion module

            # Compute gradients
            loss.backward()

            with torch.no_grad():
                # Update delta with PGD step and projection in the L2 ball
                delta.copy_(pgd_step(delta, delta.grad, self.opts.step_size))
                delta.copy_(l2_project(delta, self.opts.epsilon))
            
            if self.opts.use_fusion_module and update_weights:
                # Update the fusion module weights with the optimizer
                self.optimizer_fusion.step()

            # log the inner steps of the PGD attack
            if self.opts.log_inner_steps and i % 10 == 0:
                 self.writer.add_image(f"{tag}/Mask", mask[0], self.inner_step_count)

            if self.opts.log_inner_steps:  
                self.writer.add_scalar(f"{tag}/PGD/loss", loss.item(), self.inner_step_count)
                self.writer.add_scalar(f"{tag}/PGD/adv_loss", ladv.item(), self.inner_step_count)
                self.writer.add_scalar(f"{tag}/PGD/rec_loss", lrec.item(), self.inner_step_count)
                self.writer.add_scalar(f"{tag}/PGD/delta_l2", torch.norm(delta.detach(), dim=1).mean().item(), self.inner_step_count)
            
            self.inner_step_count += 1

            # Save the best attack in the inner loop of the PGD
            if loss.item() < best_loss:
                best_loss = loss.item()
                if self.opts.use_fusion_module:
                    best_attack = (x_final.detach(), zid.detach(), zadv.detach(), delta, loss.item(), ladv.item(), lrec.item())
                else:
                    best_attack = (x_adv.detach(), zid.detach(), zadv.detach(), delta, loss.item(), ladv.item(), lrec.item())

        # After finishing we compute the final attack with the last delta obtained, 
        # to compare it with the best attack obtained in the inner loop of the PGD
        with torch.no_grad():
            self.fusion_net.eval() # only to be sure that the fusion net is in eval mode to compute the final attack
            x_adv = self.aadblocks(inputs=(zatt, zid + delta))
            if self.opts.use_fusion_module and self.opts.use_weight_mask:
                x_final, mask = self.fusion_net(img_org, x_adv, self.spatial_weights)
            else:
                x_final, mask = self.fusion_net(img_org, x_adv)
            
            if self.opts.use_fusion_module:
                x_final_aligned = alignment(x_final)
            else:
                x_final_aligned = alignment(x_adv)
            x_final_for_net = (x_final_aligned - 0.5) / 0.5
            zadv = l2_norm(self.facenet(x_final_for_net)).detach()

            if self.opts.use_fusion_module:
                lrec = self.rec_loss(x_final, img_org) 
            else:
                lrec = self.rec_loss(x_adv, img_org)

            ladv = self.adv_loss(zadv, zid) 

            if self.opts.use_fusion_module and self.opts.use_weight_mask:
                loss_mask_reg = torch.mean(mask)
                # compute the total loss
                loss = ladv + lrec + (self.opts.mask_reg * loss_mask_reg)
            else:
                loss = ladv + lrec
        if self.opts.use_fusion_module:
            final_attack = (x_final.detach(), zid.detach(), zadv.detach(), delta, loss.item(), ladv.item(), lrec.item())
        else:
            final_attack = (x_adv.detach(), zid.detach(), zadv.detach(), delta, loss.item(), ladv.item(), lrec.item())

        if final_attack[4] < best_loss:
            return final_attack
        else:
            return best_attack

    def attack_batch_baseline_linf(self, img_wm, tag):
        """
        Baseline attack: PGD directly in the pixel space with L-infinity constraint, without using the AAD network or the fusion module.
        """
        # 1. Get the original embedding as reference (zid) before the attack
        with torch.no_grad():
            img_org_aligned = alignment(img_wm) # resize to 112x112 for ARCface
            img_org_for_net = (img_org_aligned - 0.5) / 0.5 # ARCFace normalization
            zid = l2_norm(self.facenet(img_org_for_net)).detach() # Normakization to facilitate the project step of the PGD

        # 2. Initialize delta (perturbation in the pixel space)
        #delta_img = torch.zeros_like(img_wm).to(self.device)
        delta_img = torch.zeros_like(img_wm).uniform_(-self.opts.epsilon, self.opts.epsilon).to(self.device)
        delta_img.requires_grad = True # To be able to compute gradients with respect to the perturbation in the pixel space

        best_attack = None
        best_loss = float('inf')

        # Loop PGD
        for i in range(self.opts.pgd_steps):
            # Generate the adversarial image by adding the perturbation to the original watermarked image
            x_adv = torch.clamp(img_wm + delta_img, 0, 1)

            # 3. Extract embedding of the perturbed image to compute loss
            x_adv_aligned = alignment(x_adv) # resize to 112x112 for ARCface
            x_adv_for_net = (x_adv_aligned - 0.5) / 0.5 # ARCFace normalization
            zadv = l2_norm(self.facenet(x_adv_for_net)) # Normakization to facilitate the project step of the PGD

            # 4. Compute losses
            # we want to minimize the similarity between the adversarial embedding and the original one (maximize the distance)
            ladv = self.adv_loss(zadv, zid)
            # we want to preserve the watermark and the visual quality of the image,
            lrec = self.rec_loss(x_adv, img_wm)
            loss = ladv +  lrec # final loss to minimize

            # 5. Backpropagation and PGD step
            if delta_img.grad is not None: delta_img.grad.zero_()
            loss.backward()

            # 6. PGD L-infinity step and Projection
            with torch.no_grad():
                # Update delta_img with PGD step and projection in the L-infinity ball
                delta_img.copy_(pgd_step_linf(delta_img, delta_img.grad, self.opts.step_size))
                delta_img.copy_(linf_project(delta_img, self.opts.epsilon))

            if self.opts.log_inner_steps and i % 10 == 0:
                 evident_perturbation = torch.abs(delta_img[0]) / self.opts.epsilon
                 self.writer.add_image(f"{tag}/Perturbation", evident_perturbation, self.inner_step_count)

            if self.opts.log_inner_steps:  
                self.writer.add_scalar(f"{tag}/PGD/loss", loss.item(), self.inner_step_count)
                self.writer.add_scalar(f"{tag}/PGD/adv_loss", ladv.item(), self.inner_step_count)
                self.writer.add_scalar(f"{tag}/PGD/rec_loss", lrec.item(), self.inner_step_count)
                self.writer.add_scalar(f"{tag}/PGD/delta_l2", torch.norm(delta_img.detach(), dim=1).mean().item(), self.inner_step_count)
            
            self.inner_step_count += 1

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_attack = (x_adv.detach(), zid.detach(), zadv.detach(), delta_img.detach(), 
                               loss.item(), ladv.item(), lrec.item())

        # After finishing we compute the final attack with the last delta_img obtained, 
        # to compare it with the best attack obtained in the inner loop of the PGD        
        with torch.no_grad():
            x_adv = torch.clamp(img_wm + delta_img, 0, 1)
            zadv = l2_norm(self.facenet((alignment(x_adv) - 0.5) / 0.5))
            self.ladv = self.adv_loss(zadv, zid)
            self.lrec = self.rec_loss(x_adv, img_wm)
            loss = self.ladv + self.lrec
        
        final_attack = (x_adv.detach(), zid.detach(), zadv.detach(), delta_img.detach(), loss.item(), ladv.item(), lrec.item())
        
        if final_attack[4] < best_loss:
            return final_attack
        else:
            return best_attack
        
    def attack_batch_baseline_l2_masked(self, imgs_wm, tag):

        if imgs_wm.max() > 1.0:
            imgs_wm = imgs_wm / 255.0

        B = imgs_wm.shape[0]

        with torch.no_grad():
            boxes_list, landmarks_list = self.facenet.detect_boxes_batch(imgs_wm)
            if any(b is None for b in boxes_list):
                raise ValueError("MTCNN failed to detect face in some images")
            zid = l2_norm(self.facenet.embed_batch_with_boxes(
                imgs_wm, boxes_list, landmarks_list, requires_grad=False
            )).detach()

        delta_img = torch.randn_like(imgs_wm).to(self.device)
        delta_img = l2_project(delta_img, self.opts.epsilon)  
        delta_img.requires_grad = True

        best_attack = None
        best_loss = float('inf')

        face_masks = []
        for i in range(B):
            fm = generate_face_box_mask(boxes_list[i][None], imgs_wm[i:i+1].shape, self.device)
            face_masks.append(fm)
        face_mask = torch.cat(face_masks, dim=0)
        final_mask = face_mask.to(self.device)

        for i in range(self.opts.pgd_steps):
            delta_masked = delta_img * final_mask
            x_adv = torch.clamp(imgs_wm + delta_masked, 0, 1)

            zadv = l2_norm(self.facenet.embed_batch_with_boxes(
                x_adv, boxes_list, landmarks_list, requires_grad=True))

            ladv = self.adv_loss(zadv, zid)
            lrec = self.rec_loss(x_adv, imgs_wm)
            #loss  = ladv + lrec
            lfreq = self.freq_loss(x_adv, imgs_wm)
            loss  = ladv + lrec + lfreq

            if delta_img.grad is not None:
                delta_img.grad.zero_()
            loss.backward()

            with torch.no_grad():
                grad_masked = delta_img.grad * final_mask   
                delta_img.copy_(pgd_step(delta_img, grad_masked, self.opts.step_size))
                delta_img.copy_(delta_img * final_mask)   
                delta_img.copy_(l2_project(delta_img, self.opts.epsilon))

            if self.opts.log_inner_steps:  
                delta_l2 = torch.norm(
                    delta_img.detach().view(delta_img.size(0), -1), dim=1
                ).mean().item()
                
                delta_l2_per_img = torch.norm(
                    delta_img.detach().view(delta_img.size(0), -1), dim=1
                )  # [B]
                budget_used_frac = (delta_l2_per_img / (self.opts.epsilon + 1e-12)).mean().item()
                
                delta_mag = torch.norm(delta_img[0].detach(), dim=0, keepdim=True)  # [1, H, W]
                evident_perturbation = delta_mag / (delta_mag.max() + 1e-12)
                evident_perturbation = evident_perturbation.clamp(0, 1)

                self.writer.add_image(f"{tag}/Perturbation", evident_perturbation, self.inner_step_count)
                self.writer.add_scalar(f"{tag}/PGD/loss",          loss.item(),        self.inner_step_count)
                self.writer.add_scalar(f"{tag}/PGD/adv_loss",      ladv.item(),        self.inner_step_count)
                self.writer.add_scalar(f"{tag}/PGD/rec_loss",      lrec.item(),        self.inner_step_count)
                self.writer.add_scalar(f"{tag}/PGD/freq_loss",     lfreq.item(),       self.inner_step_count)
                self.writer.add_scalar(f"{tag}/PGD/delta_l2",      delta_l2,           self.inner_step_count)
                self.writer.add_scalar(f"{tag}/PGD/budget_used",   budget_used_frac,   self.inner_step_count)

                self.per_step_loss_buffer.append(loss.item())
                self.per_step_delta_buffer.append(delta_l2) 
                self.per_step_lrec_buffer.append(lrec.item())
                self.per_step_ladv_buffer.append(ladv.item())
                self.per_step_lfreq_buffer.append(lfreq.item())
            
            self.inner_step_count += 1

            if  loss.item() < best_loss:
                best_loss = loss.item()
                best_attack = (x_adv.detach(), zid.detach(), zadv.detach(),
                            delta_img.clone().detach(), loss.item(), ladv.item(), lrec.item(), lfreq.item(), i)

        return best_attack

    def attack_batch_baseline_linf_masked(self, imgs_wm, tag):
        """
        Baseline attack: PGD directly in the pixel space with L-infinity constraint, without using the AAD network or the fusion module.
        """
        # image normalization if it in normal image range
        if imgs_wm.max() > 1.0:
            imgs_wm = imgs_wm / 255.0

        B = imgs_wm.shape[0]
        #print('batch size:', B)

        # 1. Get the original embedding as reference (zid) before the attack
        with torch.no_grad():
            # detect the face in the watermarked image to be able to do the alignment for the ARCface model
            boxes_list, landmarks_list = self.facenet.detect_boxes_batch(imgs_wm)
            #compare_crops_interactive_facenet(imgs_wm, boxes_list, landmarks_list, image_size=160)
            #compare_crops_interactive_arcface(imgs_wm, boxes_list, landmarks_list, image_size=112)
            
            if any(b is None for b in boxes_list):
                raise ValueError("MTCNN failed to detect face in some images")
            
            zid = l2_norm(self.facenet.embed_batch_with_boxes(
                                imgs_wm, boxes_list, landmarks_list, requires_grad=False
                            )).detach()

        # 2. Initialize delta (perturbation in the pixel space)
        #delta_img = torch.zeros_like(imgs_wm).to(self.device)
        delta_img = torch.zeros_like(imgs_wm).uniform_(-self.opts.epsilon, self.opts.epsilon).to(self.device)
        delta_img.requires_grad = True # To be able to compute gradients with respect to the perturbation in the pixel space

        best_attack = None
        best_loss = float('inf')

        # 3. Compute masks in batch [B, C, H, W]
        face_masks = []

        # compute the masks for the masked PGD attack
        for i in range(B):
            fm = generate_face_box_mask(boxes_list[i][None], imgs_wm[i:i+1].shape, self.device)
            face_masks.append(fm)

        # Compound binary mask: [B, C, H, W]
        # 1 = Attack, face and low frequences | 0 = Don't attack, background and high frequencies (edges)
        face_mask = torch.cat(face_masks, dim=0)    # [B, C, H, W]
        #mask_edges = compute_sobel_edges_mask(imgs_wm, threshold=0.5)    # [B, 1, H, W]
        #final_mask = (face_mask * (1 - mask_edges)).to(self.device)
        final_mask = face_mask.to(self.device)

        # Loop PGD
        for i in range(self.opts.pgd_steps):
            delta_masked = delta_img * final_mask 
            # Generate the adversarial image by adding the perturbation to the original watermarked image
            x_adv = torch.clamp(imgs_wm + delta_masked, 0, 1)

            # 3. Extract embedding of the perturbed image to compute loss
            # Do the aligenment and normalization for ARCface (the attacked FR)
            zadv = l2_norm(self.facenet.embed_batch_with_boxes(
                            x_adv, boxes_list, landmarks_list, requires_grad=True))

            # 4. Compute losses
            # we want to minimize the similarity between the adversarial embedding and the original one (maximize the distance)
            ladv = self.adv_loss(zadv, zid)
            # we want to preserve the watermark and the visual quality of the image,
            lrec = self.rec_loss(x_adv, imgs_wm)
            # we want to penalyze the frequency components of the perturbation
            lfreq = self.freq_loss(x_adv, imgs_wm)
            loss = ladv +  lrec + lfreq # final loss to minimize

            # 5. Backpropagation and PGD step
            if delta_img.grad is not None: 
                delta_img.grad.zero_()
            loss.backward()

            # 6. PGD L-infinity step and Projection
            with torch.no_grad(): 
                # Update delta_img with PGD step and projection in the L-infinity ball
                delta_img.copy_(pgd_step_linf_masked(delta_img, delta_img.grad, self.opts.step_size, final_mask))
                delta_img.copy_(linf_project(delta_img, self.opts.epsilon))
                delta_img.copy_(delta_img * final_mask)

            if self.opts.log_inner_steps:  
                delta_linf = delta_img.abs().max().item()
                delta_sat = (delta_img.abs() >= self.opts.epsilon - 1e-6).float().mean().item()
                evident_perturbation = torch.abs(delta_img[0]) / self.opts.epsilon
                self.writer.add_image(f"{tag}/Perturbation", evident_perturbation, self.inner_step_count)
                self.writer.add_scalar(f"{tag}/PGD/loss", loss.item(), self.inner_step_count)
                self.writer.add_scalar(f"{tag}/PGD/adv_loss", ladv.item(), self.inner_step_count)
                self.writer.add_scalar(f"{tag}/PGD/rec_loss", lrec.item(), self.inner_step_count)
                self.writer.add_scalar(f"{tag}/PGD/freq_loss", lfreq.item(), self.inner_step_count)
                self.writer.add_scalar(f"{tag}/PGD/delta_l2", torch.norm(delta_img.detach(), dim=1).mean().item(), self.inner_step_count)
                self.writer.add_scalar(f"{tag}/PGD/delta_linf", delta_linf, self.inner_step_count)
                self.writer.add_scalar(f"{tag}/PGD/delta_sat_frac", delta_sat, self.inner_step_count)
                self.per_step_loss_buffer.append(loss.item())
                self.per_step_delta_buffer.append(delta_img.abs().max().item())
                self.per_step_lrec_buffer.append(lrec.item())
                self.per_step_ladv_buffer.append(ladv.item())
                self.per_step_lfreq_buffer.append(lfreq.item())
            
            self.inner_step_count += 1

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_attack = (x_adv.detach(), zid.detach(), zadv.detach(), delta_img.clone().detach(), 
                               loss.item(), ladv.item(), lrec.item(), lfreq.item(), i)

        return best_attack

    def run_eval_face_recognition_v1(self, filename_results="face_recognition_results.json", epoch=0, set_name='all'):
        """
        Evaluate the face recognition performance on the watermarked and attacked images.
            - Computes the cosine similarity between the template and both the watermarked and attacked images.
        :param filename_results: The filename to save the evaluation results.
        :param epoch: The current epoch number for saving results.
        :param set_name: The dataset split to evaluate ('train', 'val', 'test' or 'all').
        """
        self.create_FR_sets(set_name) # create dataloaders for the evaluation of face recognition performance after the attack 
        threshold = self.opts.face_recognition_threshold # threshold for cosine similarity to consider a match in Face Recognition

        # depending where this function is called, we want to evaluate the performance of the face recognition in different 
        # sets (train, val or test), so we create the corresponding dataloaders for each case.
        if set_name == 'all':
            names = ['train', 'test']
            loaders = [self.face_loader_train, self.face_loader_test]
        elif set_name == 'val':
            names = ['train']
            loaders = [self.face_loader_train]
        elif set_name == 'test':
            names = ['test']
            loaders = [self.face_loader_test]
        elif set_name == 'train':
            names = ['train']
            loaders = [self.face_loader_train]

        models = {
            self.opts.facenet_mode: self.facenet, # white box
            self.opts.facenet_mode_test: self.facenet_test # black box
        }

        results = {}
        for model_name, model in models.items():
            for split_name, dataloader in zip(names, loaders):
                all_sim_wm = []
                all_sim_attacked = []
                successful_attacks = 0
                correct_wm = 0
                correct_attacked = 0
                total_samples = 0
                for template_img, wm_img, attacked_img, filename in tqdm(dataloader, desc=f"Evaluating Face Recognition on {split_name} set"):
                    wm_img = wm_img.to(self.device)
                    template_img = template_img.to(self.device)
                    attacked_img = attacked_img.to(self.device)

                    # Get the embeddings for the original image and the attacked image
                    with torch.no_grad():
                        # detect boxes and landmarks for the watermarked images
                        boxes_list, landmarks_list = model.detect_boxes_batch(wm_img)
                        #compare_crops_interactive(imgs_wm, boxes_list, landmarks_list, image_size=112)
                        # extract the landmarks and boxes for the template
                        boxes_list_t, landmarks_list_t = model.detect_boxes_batch(template_img)
                        
                        # Extract the embeddings (using aligment)
                        zid_wm = l2_norm(model.embed_batch_with_boxes(
                                            wm_img, boxes_list, landmarks_list, requires_grad=False
                                        )).detach() # facial vector for the watermarked image
                        
                        zid_template = l2_norm(model.embed_batch_with_boxes(
                                            template_img, boxes_list_t, landmarks_list_t, requires_grad=False
                                        )).detach() # facial vector for the template

                        zadv_attacked = l2_norm(model.embed_batch_with_boxes(
                                            attacked_img, boxes_list, landmarks_list, requires_grad=False
                                        )).detach() # facial vector for the attacked image

                        # compute cosine similarity between the template and the watermarked image before the attack,
                        #  and between the template and the attacked image
                        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                        sim_wm = cos(zid_template, zid_wm)
                        sim_attacked = cos(zid_template, zadv_attacked)
                        
                        # compute the predicted labels based on the threshold
                        is_recognized_wm = sim_wm > threshold
                        is_recognized_attacked = sim_attacked > threshold
                        # The attack is successful if the watermarked image is recognized (sim_wm > threshold) 
                        # and the attacked image is not recognized (sim_attacked <= threshold)
                        attack_success = is_recognized_wm & (~is_recognized_attacked)
                        # update the counters for the metrics
                        successful_attacks += attack_success.sum().item()
                        correct_wm += is_recognized_wm.sum().item()
                        correct_attacked += is_recognized_attacked.sum().item()
                        total_samples += template_img.size(0)
                        # store the cosine similarities for both cases to compute the average and std later
                        all_sim_wm.extend(sim_wm.cpu().numpy().tolist())
                        all_sim_attacked.extend(sim_attacked.cpu().numpy().tolist())

                # --- Compute final metrics for the set ---
                acc_original = (correct_wm / total_samples) * 100
                acc_attacked = (correct_attacked / total_samples) * 100
                # Attack Success Rate (ASR) is the percentage of samples where the watermarked image is correctly recognized but the attacked image is not recognized.
                asr = (successful_attacks / correct_wm) * 100 if correct_wm > 0 else 0

                print(f"\nResultados {split_name.upper()} for model {model_name}:")
                print(f"  - Acc Original: {acc_original:.2f}%")
                print(f"  - Acc Post-Ataque: {acc_attacked:.2f}%")
                print(f"  - Attack Success Rate (ASR): {asr:.2f}%")

                # compute the average cosine similarity for both cases
                avg_sim_wm = np.mean(all_sim_wm)
                avg_sim_attacked = np.mean(all_sim_attacked)
                std_sim_wm = np.std(all_sim_wm)
                std_sim_attacked = np.std(all_sim_attacked)

                print(f"Mean Cosine Similarity - Watermarked: {avg_sim_wm:.4f} ± {std_sim_wm:.4f}")
                print(f"Mean Cosine Similarity - Attacked: {avg_sim_attacked:.4f} ± {std_sim_attacked:.4f}")    
                
                results[f"{split_name}_{model_name}"] = {"acc_wm": acc_original, "asr": asr, "acc_attacked":acc_attacked,
                                    "avg_sim_wm": avg_sim_wm, "std_sim_wm": std_sim_wm, "avg_sim_attacked": avg_sim_attacked, "std_sim_attacked": std_sim_attacked,
                                    "all_sim_wm": all_sim_wm, "all_sim_attacked": all_sim_attacked}
        # save the results in a json file
        with open(os.path.join(self.output_to_save, self.folder_struct, self.id_number_exp, f'{set_name}_ep{epoch}_{filename_results}'), 'w') as f:
            json.dump(results, f, indent=4)

    def run_eval_face_recognition(self, filename_results="face_recognition_results.json", epoch=0, set_name='all'):
        self.create_FR_sets(set_name)

        if set_name == 'all':
            names = ['train', 'test']
            loaders = [self.face_loader_train, self.face_loader_test]
        elif set_name in ['val', 'train']:
            names = ['train']
            loaders = [self.face_loader_train]
        elif set_name == 'test':
            names = ['test']
            loaders = [self.face_loader_test]

        models = {
            self.opts.facenet_mode: self.facenet,
            self.opts.facenet_mode_test: self.facenet_test
        }

        results = {}
        for model_name, model in models.items():
            for split_name, dataloader in zip(names, loaders):
                all_sim_wm = []
                all_sim_attacked = []
                total_samples = 0

                for template_img, wm_img, attacked_img, filename in tqdm(dataloader, desc=f"FR Eval {split_name}"):
                    wm_img = wm_img.to(self.device)
                    template_img = template_img.to(self.device)
                    attacked_img = attacked_img.to(self.device)

                    with torch.no_grad():
                        boxes_wm, lm_wm   = model.detect_boxes_batch(wm_img)
                        boxes_t,  lm_t    = model.detect_boxes_batch(template_img)

                        zid_wm       = l2_norm(model.embed_batch_with_boxes(wm_img,       boxes_wm, lm_wm, requires_grad=False))
                        zid_template = l2_norm(model.embed_batch_with_boxes(template_img, boxes_t,  lm_t,  requires_grad=False))
                        zadv         = l2_norm(model.embed_batch_with_boxes(attacked_img, boxes_wm, lm_wm, requires_grad=False))

                        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                        sim_wm      = cos(zid_template, zid_wm)
                        sim_attacked = cos(zid_template, zadv)

                        all_sim_wm.extend(sim_wm.cpu().numpy().tolist())
                        all_sim_attacked.extend(sim_attacked.cpu().numpy().tolist())
                        total_samples += template_img.size(0)

                results[f"{split_name}_{model_name}"] = {
                    "total_samples":     total_samples,
                    "all_sim_wm":        all_sim_wm,
                    "all_sim_attacked":  all_sim_attacked,
                    "avg_sim_wm":        float(np.mean(all_sim_wm)),
                    "std_sim_wm":        float(np.std(all_sim_wm)),
                    "avg_sim_attacked":  float(np.mean(all_sim_attacked)),
                    "std_sim_attacked":  float(np.std(all_sim_attacked)),
                    "delta_sim_mean":    float(np.mean(all_sim_wm) - np.mean(all_sim_attacked))
                }

                print(f"\n[FR] {split_name.upper()} | model={model_name}")
                print(f"  avg_sim_wm:       {np.mean(all_sim_wm):.4f} ± {np.std(all_sim_wm):.4f}")
                print(f"  avg_sim_attacked: {np.mean(all_sim_attacked):.4f} ± {np.std(all_sim_attacked):.4f}")
                print(f"  delta_sim_mean:   {np.mean(all_sim_wm) - np.mean(all_sim_attacked):.4f}")

        out_path = os.path.join(self.output_to_save, self.folder_struct,
                                self.id_number_exp,
                                f'{set_name}_ep{epoch}_{filename_results}')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=4)
                    
    def run_attack(self, tag):
        """ 
        Main loop to run the attack for both training and testing.
         - For training, it iterates over epochs and batches, applies the attack, computes metrics, and updates the fusion module if specified.
         - For testing, it runs the attack without updating weights and computes metrics for evaluation.
         - If the baseline flag is True, it runs a pixel-based attack directly on the watermarked images. 
         Otherwise, it runs the complete pipeline attack on the embeddings.
        """
        epochs = 1
        if tag == 'train' and self.opts.restore_training:
            step_count = self.global_step
            start_epoch = self.start_epoch
        else:
            step_count = 0
            start_epoch = 0

        update_weights = False
        best_global_loss_avg = float('inf')
        tag_new = None
            
        if tag == 'train': 
            tag_new = 'train'
            loader = self.train_dataloader
            if self.opts.baseline == False:  # pipeline way
                epochs = self.opts.epochs
                start_epoch = self.start_epoch
                if self.opts.use_fusion_module:
                    update_weights = True
                if start_epoch >= epochs:
                    print("[*] Training already completed according to checkpoint.")
                    return {}
                
        if tag == 'test':
            if self.opts.baseline == False:  # pipeline way
                loader = self.val_dataloader
                tag_new = 'val'
                update_weights = False
                
            else:
                loader = self.test_dataloader
                tag_new = 'test'

        if tag == 'generalization' and self.opts.baseline == False:
            loader = self.test_dataloader
            tag_new = 'test'
                
        total_steps = len(loader) * epochs
        pbar_epoch = tqdm(range(start_epoch, epochs), desc=f"{tag} Epochs", position=0, leave=True, total=epochs)
        # iterations over epochs and batches ...
        for epoch in pbar_epoch:
            pbar_batch = tqdm(
                loader,
                desc=f"{tag_new} Batches (epoch {epoch+1}/{epochs})",
                position=1,
                leave=False,
                total=len(loader)
                )
            
            # Consolidate results for the epoch
            results = {'psnr_start': [], 'ssim_start': [], 'psnr': [], 'ssim': [], 
                       'acc_before': [], 'acc_after': [], 'cos_sim': [], 'loss': [],
                       'psnr_std_start': [], 'ssim_std_start': [], 'psnr_std': [], 'ssim_std': [],
                       'acc_std_before': [], 'acc_std_after': [], 'cos_sim_std': [], 'loss_std': []}
            # Loop over batches
            for i, (imgs_wm, filenames, real_wms, org_imgs) in enumerate(pbar_batch):
                imgs_wm = imgs_wm.to(self.device) # watermarked images (before attack)
                real_wms = real_wms.to(self.device) # real watermarks (binary)
                org_imgs = org_imgs.to(self.device) # original images (before watermarking)

                #print('range imgs_wm:', imgs_wm.min().item(), imgs_wm.max().item(), "and shape:", imgs_wm.shape)
                #print('range real_wms:', real_wms.min().item(), real_wms.max().item(), "and shape:", real_wms.shape)
                #print('range org_imgs:', org_imgs.min().item(), org_imgs.max().item(), "and shape:", org_imgs.shape)

                # Run the corresponding attack for the batch
                if self.opts.baseline == True: # we attack directly the watermarked images with a pixel-based attack (L-infinity)
                    #imgs_adv, zid, zadv, delta, loss, ladv, lrec, lfreq, i_pgd = self.attack_batch_baseline_linf_masked(imgs_wm, tag=tag_new)
                    imgs_adv, zid, zadv, delta, loss, ladv, lrec, lfreq, i_pgd = self.attack_batch_baseline_l2_masked(imgs_wm, tag=tag_new)
                #else: # we attack the embeddings with the complete pipeline (AAD + Fusion)
                #    imgs_adv, zid, zadv, delta, loss, ladv, lrec= self.attack_batch_pipeline(imgs_wm, tag=tag_new, update_weights=update_weights)

                if i == 0:
                    self._save_perturbations_inf(delta, prefix=f'{tag_new}_batch0')
        
                if i == 0 and self.opts.log_inner_steps:
                    for step_idx, (l, d) in enumerate(zip(
                            self.per_step_loss_buffer, 
                            self.per_step_delta_buffer,
                            )):
                        self.writer.add_scalar(f"{tag_new}/PGD_traj/loss_vs_step_batch0", l, step_idx)
                        self.writer.add_scalar(f"{tag_new}/PGD_traj/delta_vs_step_batch0", d, step_idx)
                        self.writer.add_scalar(f"{tag_new}/PGD_traj/rec_loss_vs_step_batch0", self.per_step_lrec_buffer[step_idx], step_idx)
                        self.writer.add_scalar(f"{tag_new}/PGD_traj/adv_loss_vs_step_batch0", self.per_step_ladv_buffer[step_idx], step_idx)
                        self.writer.add_scalar(f"{tag_new}/PGD_traj/freq_loss_vs_step_batch0", self.per_step_lfreq_buffer[step_idx], step_idx)
                
                if self.opts.log_inner_steps:
                    self.per_step_loss_buffer.clear()
                    self.per_step_delta_buffer.clear()
                    self.per_step_lrec_buffer.clear()
                    self.per_step_ladv_buffer.clear()
                    self.per_step_lfreq_buffer.clear()

                print("id iteration best attack:", i_pgd, "loss:", loss, "ladv:", ladv, "lrec:", lrec, "lfreq:", lfreq)
                
                # If we are training the fusion module
                if update_weights and tag_new == 'train':
                    self.global_step += 1
                    current_log_step = self.global_step
                    is_best = False
                    if loss < self.best_global_loss:
                        self.best_global_loss = loss
                        is_best = True
                    # save the model every 50 steps or if it's the best model so far
                    if (self.global_step % 50 == 0) or is_best:
                        self._save_checkpoint(epoch, loss, is_best=is_best)
                else:
                    if tag_new == 'val':
                        self.val_step_count += 1
                        current_log_step = self.val_step_count
                    else:
                        step_count += 1
                        current_log_step = step_count

                # Compute metrics and log results after the attack for the current batch
                with torch.no_grad():
                    # Cosine similarity between original and attacked embeddings
                    cos_sim = torch.cosine_similarity(zadv, zid).mean().item()

                    # Watermark
                    # we expect that the image before decoder should be in the range [0,1]
                    # all the inputs are in the range [0,1]
                    if self.opts.wm_algorithm.lower() == 'stegformer':
                        if imgs_wm.max() > 1.0:
                            imgs_wm = imgs_wm / 255.0
                        imgs_wm_c = torch.clamp(imgs_wm, 0, 1)
                        imgs_adv_c = torch.clamp(imgs_adv, 0, 1) # always in [0,1]
                    
                    elif self.opts.wm_algorithm.lower() == 'stegaformer':
                        if imgs_wm.max() <= 1.0:
                            imgs_wm_c = (imgs_wm * 255.0).clamp(0, 255)
                        else:
                            imgs_wm_c = imgs_wm   
                        imgs_adv_c = (imgs_adv * 255.0).clamp(0, 255)

                    # Recover the watermark from both the watermarked image before the attack and the attacked image
                    wm_before = self.wm_decoder(imgs_wm_c)
                    wm_after = self.wm_decoder(imgs_adv_c)
                    # compute the bit accuracy rate for both cases
                    acc_b = self.get_bit_accuracy_rate(real_wms, wm_before, bpp=self.opts.wm_bpp)
                    acc_a = self.get_bit_accuracy_rate(real_wms, wm_after, bpp=self.opts.wm_bpp)

                    #print('before attack - acc wm:', acc_b, "after attack - acc wm:", acc_a, "cosine similarity:", cos_sim, "loss:", loss)
                    # ensure to be in the same range for the computation of PSNR and SSIM
                    o_imgs, w_imgs, a_imgs = [
                                (img * 255.0 if self.data_range == 255 and img.max() <= 1.0 else 
                                img / 255.0 if self.data_range == 1 and img.max() > 1.0 else img) 
                                for img in [org_imgs, imgs_wm, imgs_adv]
                            ]
                            
                    # Compute PSNR and SSIM between the attacked image and the watermarked image before the attack
                    p_initial = self.cal_psnr(o_imgs, w_imgs).item()
                    s_initial = self.cal_ssim(o_imgs, w_imgs).item()
                    p = self.cal_psnr(a_imgs, w_imgs).item()
                    s = self.cal_ssim(a_imgs, w_imgs).item()

                    # Store results for the batch in the epoch results
                    results['psnr_start'].append(p_initial)
                    results['ssim_start'].append(s_initial)
                    results['psnr'].append(p)
                    results['ssim'].append(s)
                    results['acc_before'].append(acc_b)
                    results['acc_after'].append(acc_a)
                    results['cos_sim'].append(cos_sim)
                    results['loss'].append(loss)
                    results['psnr_std_start'].append(p_initial)
                    results['ssim_std_start'].append(s_initial)
                    results['psnr_std'].append(p)
                    results['ssim_std'].append(s)
                    results['acc_std_before'].append(acc_b)
                    results['acc_std_after'].append(acc_a)
                    results['cos_sim_std'].append(cos_sim)
                    results['loss_std'].append(loss)

                    # Store the images of the attacked samples for qualitative evaluation (in npy and png formats)
                    self.save_samples(imgs_adv, filenames, tag=tag_new)

                # Log metrics to TensorBoard every 10 steps
                if (current_log_step + 1) % 10 == 0:
                    self.writer.add_scalar(f'{tag_new}/Loss', loss, current_log_step)
                    self.writer.add_scalar(f'{tag_new}/Adv_Loss', ladv, current_log_step)
                    self.writer.add_scalar(f'{tag_new}/Rec_Loss', lrec, current_log_step)
                    self.writer.add_scalar(f'{tag_new}/Freq_Loss', lfreq, current_log_step)
                    self.writer.add_scalar(f'{tag_new}/Cosine_Similarity', cos_sim, current_log_step)
                    self.writer.add_scalar(f'{tag_new}/PSNR', p, current_log_step)
                    self.writer.add_scalar(f'{tag_new}/SSIM', s, current_log_step)
                    self.writer.add_scalar(f'{tag_new}/WM_Acc_Before', acc_b, current_log_step)
                    self.writer.add_scalar(f'{tag_new}/WM_Acc_After', acc_a, current_log_step)
                
                #step_count += 1 
                pbar_batch.set_postfix(step=f"{current_log_step}/{total_steps}")
            
            if update_weights and tag_new == 'train': # only step the scheduler at the end of each epoch if we are training the fusion module
                self.scheduler.step()
                self._save_checkpoint(epoch + 1, loss, is_best=False)  

            epoch_avg_loss = np.mean(results['loss']) # for multiple epochs
            if epoch_avg_loss < best_global_loss_avg:
                best_average_results = {k: (np.std(v) if 'std' in k else np.mean(v)) for k, v in results.items()}
                best_global_loss_avg = epoch_avg_loss

            # after each epoch we do the evaluation of face recognition performance.
            if self.opts.baseline == False and tag_new == 'train': # only for pipeline way,
                # after each epoch of training we evaluate the face recognition performance on the validation set
                self.testing(f'val_results_{epoch + 1}.json')
                self.testing(f'test_results_{epoch + 1}.json', tag='generalization') # testing generalization on the second dataset
                self.run_eval_face_recognition(epoch=epoch + 1, set_name='val') 
                self.run_eval_face_recognition(epoch=epoch + 1, set_name='test') 

        return best_average_results # Return the best average results

    def save_samples(self, imgs_adv, filenames, tag):
        """
        This function saves the attacked images in both PNG and NPY formats for qualitative evaluation.
        The images are stored in a structured directory based on the experiment ID and the tag (train/test). 
        The NPY files contain the raw tensor data, while the PNG files are for visual inspection.
        """
        for i in range(len(imgs_adv)):
            # 1. We change the order to [256, 256, 3] (H, W, C)
            data_for_numpy = imgs_adv[i].permute(1, 2, 0)
            name = f"{filenames[i].split('.')[0]}.png"
            # Store as png and npy files
            np.save(os.path.join(self.imgout_dir, 
                                 tag,
                                 name.replace('.png', '.npy')), 
                    data_for_numpy.detach().cpu().numpy().astype(np.float32))
            # Save the attacked image as PNG for visual evaluation
            img_to_save = tensor2img(imgs_adv[i])
            img_to_save.save(os.path.join(self.imgout_dir, 
                                 tag,
                                 name))

    def _save_perturbations_inf(self, delta, prefix='delta', max_images=8):
        """
        Save the perturbation patterns (delta) for a batch of images in both visual and numerical formats for analysis.
        delta: [B, C, H, W] range [-ε, +ε]
        """
        
        B = min(delta.shape[0], max_images)  # máximo 8 imágenes
        
        for idx in range(B):
            delta_img = delta[idx]  # [C, H, W]
            delta_scaled = (delta_img - delta_img.min()) / (delta_img.max() - delta_img.min() + 1e-8)
            
            delta_mag = torch.sqrt((delta_img ** 2).sum(dim=0))  # [H, W]
            delta_mag_norm = delta_mag / (delta_mag.max() + 1e-8)  # normalizar a [0,1]

            vutils.save_image(delta_scaled, 
                            os.path.join(self.delta_dir, f'{prefix}_img{idx:02d}_pattern.png'))
            
            delta_mag_np = (delta_mag_norm.cpu().numpy() * 255).astype('uint8')
            Image.fromarray(delta_mag_np).save(
                os.path.join(self.delta_dir, f'{prefix}_img{idx:02d}_magnitude.png'))
            
    def save_params_to_json(self):
        """
        Store all the hyperparameters and settings of the current experiment 
        in a JSON file for future reference and reproducibility.
        """
        save_path = os.path.join(self.output_to_save, self.folder_struct, 
                                 self.id_number_exp, 'hyperparameters.json')
        try:
            # Convertimos el Namespace de argumentos a diccionario
            params_dict = vars(self.opts)
            
            with open(save_path, 'w') as f:
                # indent=4 hace que sea legible por humanos
                json.dump(params_dict, f, indent=4)
                    
            print(f"[*] Hyperparameters saved to {save_path}")
            
        except Exception as e:
            print(f"[!] Error saving hyperparameters JSON: {e}")

    def load_params_from_json(self):
        """
        Load hyperparameters and settings from a JSON file and update the current experiment's options.
        This is useful for resuming experiments or ensuring consistency across runs.
        """
        load_path = os.path.join(self.output_to_save, self.folder_struct, 
                                 self.id_number_exp, 'hyperparameters.json')
        try:
            with open(load_path, 'r') as f:
                params_dict = json.load(f)
            
            # Update the current options with the loaded parameters
            for key, value in params_dict.items():
                setattr(self.opts, key, value)
                    
            print(f"[*] Hyperparameters loaded from {load_path}")
            
        except Exception as e:
            print(f"[!] Error loading hyperparameters JSON: {e}")

    def get_bit_accuracy_rate(self,
        msg: torch.Tensor,
        deco_msg: torch.Tensor,
        bpp: int = 1,
        wm_algorithm: str = 'StegFormer'
    ) -> float:
        if wm_algorithm.lower() == 'stegformer':
            pixel_acc = get_message_accuracy_StegFormer(msg, deco_msg, bpp=bpp)
        elif wm_algorithm.lower() == 'stegaformer':
            pixel_acc = get_message_accuracy_StegaFormer(msg, deco_msg, 16)
        else:
            raise ValueError(f"Unknown wm_algorithm: {wm_algorithm}")


        return pixel_acc

    def training(self, tag='train'):
        print(f"[*] Starting Attack on {self.opts.dataset}")
        train_results = self.run_attack(tag=tag)
        print(f"[*] Training Results: {train_results}")

        # save the best results in a json file for later reference
        train_results_path = os.path.join(self.output_to_save, self.folder_struct, 
                                 self.id_number_exp, 'train_results_last_epoch.json')
        with open(train_results_path, 'w') as f:
            json.dump(train_results, f, indent=4)

        # run the testing evaluation
        # for baseline it is the second dataset
        # for pipeline way it is the templates of the same dataset that we are attacking (validation set)
        if self.opts.baseline == True:
            self.testing('test_results.json')
        
    def testing(self, name_file, tag='test'):    
        # Test de Generalización
        print(f"[*] Testing Generalization")
        test_results = self.run_attack(tag=tag)
        print(f"[*] Test Results: {test_results}")

        # save the test results in a json file for later reference
        test_results_path = os.path.join(self.output_to_save, self.folder_struct, 
                                 self.id_number_exp, name_file)
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=4)

    def create_sets(self):
        """
        Configures DataLoaders for training, testing, and generalization.
        """

        # DATASET CONFIGURATION LOGIC:
        # -------------------------------------------------------------------------
        self.train_dataloader = None
        self.test_dataloader = None
        self.val_dataloader = None

        # if baseline is not selected, is it neccesary
        # to split data into train, validation and test sets.
        
        # 1. Part
        # every dataset folder has templestes and test folders
        # templates are used to testing facial recognition part
        # so the identities inside the templates will be used for testing generalization
        # the rest of the identities will be used for training
        # We will use the templates photos without watermark for testing FR part
        templates_path = os.path.join(self.opts.db_path, self.opts.dataset, 'processed' , 'templates')
        # listing all the photos names inside the templates folder
        identities_templates = set([i.split('.')[0] for i in os.listdir(templates_path)]) # example: (BF0001, BF0002, ...)
        # listing the identities inside the test folder
        test_path = os.path.join(self.opts.db_path, self.opts.dataset, 'processed' , 'test')
        identities_test = set(os.listdir(test_path)) # example: (BF0001_01_N.png, BF0001_02_N.png, ...)
        
        # 2. Part
        # Now for the generalization set there is not training set only test, so we will use templates identities for validation
        # FR but all identites for testing the attack generalization and compute similarity metrics
        test_path_generalization = os.path.join(self.opts.db_path_test, self.opts.dataset_test, 'processed' , 'test')
        identities_test_generalization_all = set(os.listdir(test_path_generalization)) 

        # 3. Part
        # Create the dataloaders for train, test and generalization sets based on the identities selected
        if self.opts.baseline == False:
            # the rest between test and templates will be used for training
            identities_train = {tst.split('.')[0] for tst in identities_test if tst.split('_')[0] not in identities_templates}
            self.train_dataloader = DataLoader(
                WatermarkedDataset(self.opts.data_path, 
                                self.opts.db_path, 
                                self.opts.db_name, 
                                dataset=self.opts.dataset, 
                                identities = identities_train,
                                train_dataset=self.opts.train_dataset, 
                                wm_algorithm=self.opts.wm_algorithm, 
                                experiment_name=self.opts.experiment_name, 
                                image_size=(256, 256),
                                IMG_EXTENSION=self.opts.img_extension,
                                max_images=len(identities_train),
                                ori_data_path=self.opts.db_path),
                batch_size=self.opts.batch_size, shuffle=self.opts.train_shuffle, num_workers=self.opts.num_workers
            )
            # for validation we will use the templates identities, to evaluate the face recognition performance on the templates after each epoch of training
            identities_ttemplates = {tst.split('.')[0] for tst in identities_test if tst.split('_')[0] in identities_templates}
            self.val_dataloader = DataLoader(
                WatermarkedDataset(self.opts.data_path, 
                                self.opts.db_path, 
                                self.opts.db_name, 
                                dataset=self.opts.dataset, 
                                identities = identities_ttemplates,
                                train_dataset=self.opts.train_dataset, 
                                wm_algorithm=self.opts.wm_algorithm, 
                                experiment_name=self.opts.experiment_name, 
                                image_size=(256, 256),
                                IMG_EXTENSION=self.opts.img_extension,
                                max_images=self.opts.max_images_templates_train,
                                ori_data_path=self.opts.db_path),
                batch_size=self.opts.batch_size_test, shuffle=False, num_workers=self.opts.num_workers
            )  

        if self.opts.baseline == True: 
            # for the baseline we do not need a validation set, we will test directly on the second dataset, 
            # so we will use all the identities of the first dataset for training
            self.train_dataloader = DataLoader(
                WatermarkedDataset(self.opts.data_path, 
                                self.opts.db_path, 
                                self.opts.db_name, 
                                dataset=self.opts.dataset, 
                                identities = [tst.split('.')[0] for tst in identities_test],
                                train_dataset=self.opts.train_dataset, 
                                wm_algorithm=self.opts.wm_algorithm, 
                                experiment_name=self.opts.experiment_name, 
                                image_size=(256, 256),
                                IMG_EXTENSION=self.opts.img_extension,
                                max_images=self.opts.max_images_train,
                                ori_data_path=self.opts.db_path),
                batch_size=self.opts.batch_size_test, shuffle=False, num_workers=self.opts.num_workers
            )

        # For testing generalization
        self.test_dataloader = DataLoader(
                WatermarkedDataset(self.opts.data_path, 
                                self.opts.db_path_test, 
                                self.opts.db_name_test, 
                                dataset=self.opts.dataset_test, 
                                identities = [tst.split('.')[0] for tst in identities_test_generalization_all],
                                train_dataset=self.opts.train_dataset,
                                wm_algorithm=self.opts.wm_algorithm,
                                experiment_name=self.opts.experiment_name_test,
                                image_size=(256, 256),
                                IMG_EXTENSION=self.opts.img_extension_test,
                                max_images=self.opts.max_images_test,
                                ori_data_path=self.opts.db_path_test),
                batch_size=self.opts.batch_size_test, shuffle=False, num_workers=self.opts.num_workers
            )  
        
    def create_FR_sets(self, set_name='all'):
        """
        Creates DataLoaders for evaluating face recognition performance on both the training and testing datasets
        after the attack.
        """
        self.face_loader_train = None
        self.face_loader_test = None
        if set_name == 'all':
            print(f"[*] Creating DataLoaders for Face Recognition Evaluation on both Train and Test sets")
            # Dataloaders for the face recognition evaluation part, we will use the templates of both datasets
            self.face_loader_train = DataLoader(
                FaceAttackedDataset(self.opts.data_path,
                                    (256, 256),  
                                    self.opts.dataset,
                                    self.opts.train_dataset,
                                    self.opts.wm_algorithm, 
                                    self.opts.experiment_name, 
                                    IMG_EXTENSION=self.opts.img_extension,
                                    max_images=self.opts.max_images_templates_train,
                                    ori_data_path=self.opts.db_path,
                                    set_name='train',
                                    experiment_dir=self.opts.exp_dir,
                                    experiment_name_attack=self.folder_struct,
                                    id_experiment_attack=self.id_number_exp,
                                    face_model_attacked=self.opts.facenet_mode,
                                    attacked_dataset=self.opts.dataset))

            self.face_loader_test = DataLoader(
                FaceAttackedDataset(self.opts.data_path,
                                    (256, 256),  
                                    self.opts.dataset_test,
                                    self.opts.train_dataset,
                                    self.opts.wm_algorithm, 
                                    self.opts.experiment_name, 
                                    IMG_EXTENSION=self.opts.img_extension,
                                    max_images=self.opts.max_images_templates_test,
                                    ori_data_path=self.opts.db_path,
                                    set_name='test',
                                    experiment_dir=self.opts.exp_dir,
                                    experiment_name_attack=self.folder_struct,
                                    id_experiment_attack=self.id_number_exp,
                                    face_model_attacked=self.opts.facenet_mode,
                                    attacked_dataset=self.opts.dataset))  
        elif set_name == 'val' or set_name == 'train':
            print(f"[*] Creating DataLoader for Face Recognition Evaluation on the Validation set")
            self.face_loader_train = DataLoader(
                FaceAttackedDataset(self.opts.data_path,
                                    (256, 256),  
                                    self.opts.dataset,
                                    self.opts.train_dataset,
                                    self.opts.wm_algorithm, 
                                    self.opts.experiment_name, 
                                    IMG_EXTENSION=self.opts.img_extension,
                                    max_images=self.opts.max_images_templates_train,
                                    ori_data_path=self.opts.db_path,
                                    set_name='val',
                                    experiment_dir=self.opts.exp_dir,
                                    experiment_name_attack=self.folder_struct,
                                    id_experiment_attack=self.id_number_exp,
                                    face_model_attacked=self.opts.facenet_mode,
                                    attacked_dataset=self.opts.dataset))
        elif set_name == 'test':
            print(f"[*] Creating DataLoader for Face Recognition Evaluation on the Test set")
            self.face_loader_test = DataLoader(
                FaceAttackedDataset(self.opts.data_path,
                                    (256, 256),  
                                    self.opts.dataset_test,
                                    self.opts.train_dataset,
                                    self.opts.wm_algorithm, 
                                    self.opts.experiment_name, 
                                    IMG_EXTENSION=self.opts.img_extension,
                                    max_images=self.opts.max_images_templates_test,
                                    ori_data_path=self.opts.db_path,
                                    set_name='test',
                                    experiment_dir=self.opts.exp_dir,
                                    experiment_name_attack=self.folder_struct,
                                    id_experiment_attack=self.id_number_exp,
                                    face_model_attacked=self.opts.facenet_mode,
                                    attacked_dataset=self.opts.dataset))

def main():
    # loading the parameters
    opts = InjectionOptions().parse()
    wm_model_args = InjectionOptions.get_wm_model_args(opts)

    # Creathe base paths
    opts.output_dir = os.path.join(opts.exp_dir, f"{opts.wm_algorithm}")
    opts.output_to_save = os.path.join(opts.output_dir, 
                                       f"{opts.experiment_name}", 
                                       f"{opts.train_dataset}", 
                                       f"{opts.dataset}", 
                                       f"{opts.facenet_mode}")
    
    opts.wm_model_path = os.path.join(opts.wm_model_path,
                                      opts.wm_algorithm,
                                      'runs',
                                      opts.experiment_name,
                                      opts.train_dataset,
                                      'model')

    os.makedirs(opts.output_to_save, exist_ok=True)
    attack = AttackEmbeddings(opts, wm_model_args)

    if opts.baseline == False:  # pipeline way
        attack.training() # training, validation and testing
        
    else: # baseline way    
        if opts.only_face_recognition_evaluation:
            attack.run_eval_face_recognition('face_recognition_results_onlyfr.json')
        else:
            attack.training() # run "train" on the first dataset and testing on the second one
            # At the end we evaluate the face recognition performance
            attack.run_eval_face_recognition('face_recognition_results.json')


if __name__ == '__main__':
    main()