from argparse import ArgumentParser, Namespace

class InjectionOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        # Configuración General
        self.parser.add_argument('--seed', default=42, type=int)
        self.parser.add_argument('--device', default='cuda:0', type=str)
        self.parser.add_argument('--exp_dir', default='./output/attacks/adversarial/attack_embeddings', type=str)
        
        # Parámetros del Ataque (PGD en Embedding)
        self.parser.add_argument('--pgd_steps', default=80, type=int)
        self.parser.add_argument('--log_inner_steps', default=True, type=bool)
        self.parser.add_argument('--epsilon', default=2, type=float, help="Constraint value for delta")
        self.parser.add_argument('--step_size', default=0.03125, type=float)
        self.parser.add_argument('--epochs', default=5, type=int)
        self.parser.add_argument('--baseline', default=False, type=bool, help="If true, runs baseline")
        
        # Pesos de Pérdida
        self.parser.add_argument('--adv_weight', default=40.0, type=float, help="Weight for evasion")
        self.parser.add_argument('--rec_weight', default=10.0, type=float, help="Weight to preserve watermark/visuals")
        self.parser.add_argument('--lpips_weight', default=5.0, type=float)
        self.parser.add_argument('--mse_weight', default=1.0, type=float)
        self.parser.add_argument('--recloss_mode', default='combined', choices=['l2', 'lpips', 'combined'])
        self.parser.add_argument('--mask_reg', default=0.1, type=float, help="Weight for mask regularization")
        
        # Dataset Train (Dataset marcado)
        self.parser.add_argument('--data_path', default='/app/output/watermarking', type=str)
        self.parser.add_argument('--db_path', default='/app/facial_data', type=str)
        self.parser.add_argument('--dataset', default='CFD', type=str) # dataset of training the attacks
        self.parser.add_argument('--wm_algorithm', default='StegFormer', type=str)
        self.parser.add_argument('--train_dataset', default='celeba_hq', type=str) # this was the dataset for training the watermarking model
        self.parser.add_argument('--experiment_name', default='1_1_clamp_StegFormer-B_baseline', type=str)
        self.parser.add_argument('--db_name', default='watermarks_BBP_1_65536_500.db', type=str)
        self.parser.add_argument('--img_extension', default='npy', type=str)
        self.parser.add_argument('--max_images_train', default=831, type=int)
        self.parser.add_argument('--max_images_templates_train', default=158, type=int)
        self.parser.add_argument('--train_shuffle', default=True, type=bool)
        self.parser.add_argument('--use_fusion_module', default=True, type=bool, help="Whether to include the fusion module in the attack pipeline")
        self.parser.add_argument('--use_weight_mask', default=True, type=bool, help="Whether to include the weight mask in the attack pipeline")
        self.parser.add_argument('--lr_fusion', default=1e-4, type=float)
        self.parser.add_argument('--restore_training', default=True, type=bool, help="Whether to restore the training from a checkpoint")
        
        # Dataset Test (Generalización)
        self.parser.add_argument('--dataset_test', default='facelab_london', type=str)
        self.parser.add_argument('--db_path_test', default='/app/facial_data', type=str)
        self.parser.add_argument('--db_name_test', default='watermarks_BBP_1_65536_500.db', type=str)
        self.parser.add_argument('--experiment_name_test', default='1_1_clamp_StegFormer-B_baseline', type=str)
        self.parser.add_argument('--img_extension_test', default='npy', type=str)
        self.parser.add_argument('--max_images_test', default=102, type=int)
        self.parser.add_argument('--max_images_templates_test', default=102, type=int)

        # Modelos Pre-entrenados (Rutas)
        self.parser.add_argument('--facenet_mode', default='arcface', type=str)
        self.parser.add_argument('--facenet_dir', default='./weights/model_ir_se50.pth', type=str)
        self.parser.add_argument('--face_recognition_threshold', default=0.3, type=float, help="Threshold for cosine similarity in face recognition evaluation")
        self.parser.add_argument('--aadblocks_dir', default='./weights/AAD_best.pth', type=str)
        self.parser.add_argument('--attencoder_dir', default='./weights/Att_best.pth', type=str)
        self.parser.add_argument('--wm_model_path', default='/app/watermarking', type=str)

        # DataLoader
        self.parser.add_argument('--batch_size', default=2, type=int)
        self.parser.add_argument('--batch_size_test', default=2, type=int)
        self.parser.add_argument('--num_workers', default=1, type=int)
        self.parser.add_argument('--test_interval', default=50, type=int)

# --- PARÁMETROS ESPECÍFICOS DEL MODELO DE WATERMARKING ---
        self.parser.add_argument('--wm_use_model', default='StegFormer-B', help='Type of the model to use for watermarking')
        self.parser.add_argument('--wm_image_size', default=256, type=int, help='Size of input images for the watermarking model')
        self.parser.add_argument('--wm_bpp', default=1, type=int, help='Bits per pixel (bpp)')
        self.parser.add_argument('--wm_secret_channels', default=1, type=int, help='Canales secretos (basado en BPP)')
        self.parser.add_argument('--wm_tag', default='acc', choices=['psnr', 'ssim', 'acc', 'last'], help='Tag for loading watermark model weights')

    def parse(self):
        opts = self.parser.parse_args()
        return opts
    
    @staticmethod
    def get_wm_model_args(opts):
        """
        Extrae y empaqueta los argumentos específicos para build_StegFormer_models.
        Esto permite que la función reciba un objeto con los nombres que espera.
        """
        return Namespace(
            use_model=opts.wm_use_model,
            image_size=opts.wm_image_size,
            bpp=opts.wm_bpp,
            secret_channels=opts.wm_secret_channels,
            tag=opts.wm_tag
        )