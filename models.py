import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================================================================
# 1. 网络基础结构
# ===================================================================

class Encoder(nn.Module):
    def __init__(self, dims):
        super(Encoder, self).__init__()
        self.dims = dims
        models = []
        for i in range(len(self.dims) - 1):
            models.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            if i != len(self.dims) - 2:
                models.append(nn.ReLU())
            else:
                models.append(nn.Dropout(p=0.5))
                models.append(nn.Softmax(dim=1))
        self.models = nn.Sequential(*models)

    def forward(self, X):
        return self.models(X)


class Decoder(nn.Module):
    def __init__(self, dims):
        super(Decoder, self).__init__()
        self.dims = dims
        models = []
        for i in range(len(self.dims) - 1):
            models.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            if i == len(self.dims) - 2:
                models.append(nn.Dropout(p=0.5))
                models.append(nn.Sigmoid())
            else:
                models.append(nn.ReLU())
        self.models = nn.Sequential(*models)

    def forward(self, X):
        return self.models(X)


class GMAE_MVC(nn.Module):
    def __init__(self, input_dims, view_num, out_dims, h_dims, num_classes=10):
        super().__init__()
        self.input_dims = input_dims
        self.view_num = view_num
        self.out_dims = out_dims
        self.h_dims = h_dims
        self.num_classes = num_classes

        h_dims_reverse = list(reversed(h_dims))

        self.encoders_class_view = nn.ModuleList()
        self.encoders_non_class = nn.ModuleList()
        self.decoders_specific = nn.ModuleList()

        for v in range(self.view_num):
            self.encoders_class_view.append(Encoder([input_dims[v]] + h_dims + [out_dims]))
            self.encoders_non_class.append(Encoder([input_dims[v]] + h_dims + [out_dims]))
            self.decoders_specific.append(Decoder([out_dims * 2] + h_dims_reverse + [input_dims[v]]))

    def forward(self, x_list):
        z_c_views = []
        z_nc_list = []
        recs = []

        for v in range(self.view_num):
            x = x_list[v]
            z_c_v = self.encoders_class_view[v](x)
            z_nc_v = self.encoders_non_class[v](x)

            z_c_views.append(z_c_v)
            z_nc_list.append(z_nc_v)

        log_z_c = torch.stack([torch.log(z + 1e-8) for z in z_c_views], dim=0).mean(dim=0)
        z_c = F.softmax(log_z_c, dim=-1)

        for v in range(self.view_num):
            hidden_v = torch.cat((z_c, z_nc_list[v]), dim=-1)
            rec = self.decoders_specific[v](hidden_v)
            recs.append(rec)

        hidden = torch.cat([z_c] + z_nc_list, dim=-1)

        return z_c, z_nc_list, hidden, recs, z_c_views