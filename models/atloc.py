import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from models.att import AttentionBlock
from models.model_utils import get_mlp
from models.custom_batch_norm import CustomBatchNorm

class ScaleProcess(nn.Module):
    def __init__(self, num_bins, droprate):
        super(ScaleProcess, self).__init__()
        self.num_bins = num_bins
        self.mlp = get_mlp(num_bins, [16, 1], droprate)[0]

    def forward(self, x):
        hist = torch.stack([torch.histc(t, bins = self.num_bins, min = 0., max = 1.) for t in torch.unbind(x)], dim = 0)
        w = self.mlp(hist)
        w = w.view(-1, 1, 1, 1)
        return x * w

class FourDirectionalLSTM(nn.Module):
    def __init__(self, seq_size, origin_feat_size, hidden_size):
        super(FourDirectionalLSTM, self).__init__()
        self.feat_size = origin_feat_size // seq_size
        self.seq_size = seq_size
        self.hidden_size = hidden_size
        self.lstm_rightleft = nn.LSTM(self.feat_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.lstm_downup = nn.LSTM(self.seq_size, self.hidden_size, batch_first=True, bidirectional=True)

    def init_hidden_(self, batch_size, device):
        return (torch.randn(2, batch_size, self.hidden_size).to(device),
                torch.randn(2, batch_size, self.hidden_size).to(device))

    def forward(self, x):
        batch_size = x.size(0)
        x_rightleft = x.view(batch_size, self.seq_size, self.feat_size)
        x_downup = x_rightleft.transpose(1, 2)
        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)
        _, (hidden_state_lr, _) = self.lstm_rightleft(x_rightleft, hidden_rightleft)
        _, (hidden_state_ud, _) = self.lstm_downup(x_downup, hidden_downup)
        hlr_fw = hidden_state_lr[0, :, :]
        hlr_bw = hidden_state_lr[1, :, :]
        hud_fw = hidden_state_ud[0, :, :]
        hud_bw = hidden_state_ud[1, :, :]
        return torch.cat([hlr_fw, hlr_bw, hud_fw, hud_bw], dim=1)

class PoseNet(nn.Module):
    def __init__(self, feature_extractor, output_dim = 6, droprate = 0.5, n_channels = 1):
        super(PoseNet, self).__init__()
        self.feature_extractor = feature_extractor
        if n_channels != 3:
            self.feature_extractor.conv1 = nn.Conv2d(n_channels, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        mlps, o_dim = get_mlp(fe_out_planes, [512,], droprate)
        self.feature_extractor.fc = mlps
        self.fc = nn.Linear(o_dim, output_dim)
    def forward(self, x, get_encode = False, return_both = False):
        x = self.feature_extractor(x)
        out = self.fc(x)
        if get_encode:
            if return_both:
                return x, out
            else:
                return x
        else:
            return out


class AtLoc(nn.Module):
    def __init__(self, feature_extractor, output_dim = 6, reg_dim = 1, droprate=0.5, scale_num_bins = 30, batchnorm = False, custombn = True, pretrained=True, feat_dim=2048, n_channels = 1, lstm=False):
        super(AtLoc, self).__init__()
        self.droprate = droprate
        self.lstm = lstm

        if scale_num_bins > 0:
            self.scale_process = ScaleProcess(scale_num_bins, droprate)
        else:
            self.scale_process = None

        if batchnorm:
            self.batch_norm_1 = nn.BatchNorm2d(n_channels) if not custombn else CustomBatchNorm(n_channels, 2)
        else:
            self.batch_norm_1 = None

        # replace the last FC layer in feature extractor
        self.feature_extractor = feature_extractor
        if hasattr(self.feature_extractor, "fc"):
            if n_channels != 3:
                self.feature_extractor.conv1 = nn.Conv2d(n_channels, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
            self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
            fe_out_planes = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = nn.Sequential(nn.Linear(fe_out_planes, feat_dim), nn.Dropout(droprate), nn.LeakyReLU(0.2))
        else:
            if n_channels != 3:
                self.feature_extractor.features[0][0] = nn.Conv2d(n_channels, 96, kernel_size = 4, stride = 4)
            fe_out_planes = self.feature_extractor.head.in_features
            self.feature_extractor.head = nn.Sequential(nn.Linear(fe_out_planes, feat_dim), nn.Dropout(droprate), nn.LeakyReLU(0.2))


        if batchnorm:
            self.batch_norm_2 = nn.BatchNorm1d(feat_dim) if not custombn else CustomBatchNorm(feat_dim, 1)
        else:
            self.batch_norm_2 = None

        if self.lstm:
            self.lstm4dir = FourDirectionalLSTM(seq_size=32, origin_feat_size=feat_dim, hidden_size=256)
            fc_in_dim = feat_dim // 2
        else:
            self.att = AttentionBlock(feat_dim)
            fc_in_dim = feat_dim

        """
        self.fc_xyz = nn.Linear(fc_in_dim, 3)
        self.fc_wpqr = nn.Linear(fc_in_dim, 3)
        """
        self.fc_cls = nn.Linear(fc_in_dim, output_dim)
        self.fc_reg = nn.Sequential(nn.Linear(fc_in_dim, reg_dim), nn.Sigmoid())

        # initialize
        if pretrained:
            init_modules = [self.feature_extractor.fc, self.fc_cls, self.fc_reg]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x, get_encode = False, return_both = False):
        if self.scale_process is not None:
            x = self.scale_process(x)

        if self.batch_norm_1 is not None:
            x = self.batch_norm_1(x)
        x = self.feature_extractor(x)
        if self.batch_norm_2 is not None:
            x = self.batch_norm_2(x)

        if self.lstm:
            x = self.lstm4dir(x)
        else:
            x = self.att(x.view(x.size(0), -1))

        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)

        """
        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        out = torch.cat((xyz, wpqr), 1)
        """
        out_cls = self.fc_cls(x)
        out_reg = self.fc_reg(x)
        if get_encode:
            if return_both:
                return x, out_cls, out_reg
            else:
                return x
        else:
            return out_cls, out_reg

class AtLocPlus(nn.Module):
    def __init__(self, atlocplus):
        super(AtLocPlus, self).__init__()
        self.atlocplus = atlocplus

    def forward(self, x):
        s = x.size()
        x = x.view(-1, *s[2:])
        poses = self.atlocplus(x)
        poses = poses.view(s[0], s[1], -1)
        return poses
