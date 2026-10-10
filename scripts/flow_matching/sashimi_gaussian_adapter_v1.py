"""Paper-informed Gaussian AR wrapper around unmodified LS4 SaShiMi backbone."""
import math


def backbone_parameters(profile):
    if profile not in ('paper_flat4', 'ls4_decoder_pool1_control'):
        raise ValueError('Unknown explicitly reconstructed SaShiMi architecture')
    return dict(d_input=1, aux_channels=0, d_output=1, d_model=64, d_state=64,
        n_layers=4, d_temb=0, bidirectional=False, dropout=0., backbone='autoreg',
        use_unet=False, pool=[] if profile=='paper_flat4' else [1], expand=2, ff=2,
        s4_type='s4', use_latent=False, latent_type='none', aux_out=0, lr=.001)


def make_model(torch, native_model, config):
    class GaussianAR(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone=native_model(**config['backbone'])
            self.activation=torch.nn.Sigmoid() if config['activation']=='sigmoid' else torch.nn.Identity()
            self.sigma=config['sigma']

        def conditional_mean(self, x):
            # mu_t depends only on x_<t, including an explicit zero start token.
            shifted=torch.cat([torch.zeros_like(x[:,:1]), x[:,:-1]],dim=1)
            return self.activation(self.backbone(shifted))

        def forward(self, x):
            mu=self.conditional_mean(x)
            return (((x-mu)/self.sigma)**2*.5 + math.log(self.sigma)+.5*math.log(2*math.pi)).mean()

        def setup_rnn(self, mode='dense'):
            for module in self.backbone.modules():
                if hasattr(module,'setup_step'):module.setup_step(mode=mode)

        def recurrent_means(self, x):
            state=self.backbone.default_state(len(x),device=x.device)
            previous=torch.zeros_like(x[:,0]);means=[]
            for index in range(x.shape[1]):
                mean,state=self.backbone.step(previous,state=state)
                means.append(self.activation(mean));previous=x[:,index]
            return torch.stack(means,dim=1)

        @torch.no_grad()
        def generate(self, count, length, device='cuda'):
            previous=torch.zeros(count,1,device=device)
            state=self.backbone.default_state(count,device=device);samples=[]
            for _ in range(length):
                mean,state=self.backbone.step(previous,state=state)
                previous=self.activation(mean)+self.sigma*torch.randn_like(mean)
                samples.append(previous)
            return torch.stack(samples,dim=1)

    return GaussianAR()
