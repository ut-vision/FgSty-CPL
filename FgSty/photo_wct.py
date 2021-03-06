import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from models import VGGEncoder, VGGDecoder


class PhotoWCT(nn.Module):
    def __init__(self, gpu_id=0, fg_only=False):
        super(PhotoWCT, self).__init__()
        self.e1 = VGGEncoder(1)
        self.d1 = VGGDecoder(1)
        self.e2 = VGGEncoder(2)
        self.d2 = VGGDecoder(2)
        self.e3 = VGGEncoder(3)
        self.d3 = VGGDecoder(3)
        self.e4 = VGGEncoder(4)
        self.d4 = VGGDecoder(4)
        self.gpu_id = gpu_id
        self.fg_only = fg_only
    
    def transform(self, cont_img, styl_img, cont_seg, styl_seg):
        self.__compute_label_info(cont_seg, styl_seg)

        sF4, sF3, sF2, sF1 = self.e4.forward_multiple(styl_img)

        cF4, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3 = self.e4(cont_img)
        sF4 = sF4.data.squeeze(0)
        cF4 = cF4.data.squeeze(0)
        csF4 = self.__feature_wct(cF4, sF4, cont_seg, styl_seg)
        Im4 = self.d4(csF4, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3)

        cF3, cpool_idx, cpool1, cpool_idx2, cpool2 = self.e3(Im4)
        sF3 = sF3.data.squeeze(0)
        cF3 = cF3.data.squeeze(0)
        csF3 = self.__feature_wct(cF3, sF3, cont_seg, styl_seg)
        Im3 = self.d3(csF3, cpool_idx, cpool1, cpool_idx2, cpool2)

        cF2, cpool_idx, cpool = self.e2(Im3)
        sF2 = sF2.data.squeeze(0)
        cF2 = cF2.data.squeeze(0)
        csF2 = self.__feature_wct(cF2, sF2, cont_seg, styl_seg)
        Im2 = self.d2(csF2, cpool_idx, cpool)

        cF1 = self.e1(Im2)
        sF1 = sF1.data.squeeze(0)
        cF1 = cF1.data.squeeze(0)
        csF1 = self.__feature_wct(cF1, sF1, cont_seg, styl_seg)
        Im1 = self.d1(csF1)
        return Im1

    def _feature_interpolation(self, feat1, feat2, comb_r):
        new_feat = []
        for i in range(feat1.size()[0]):
            new_feat_per_instace = comb_r * feat1[i] + (1-comb_r) * feat2[i]
            new_feat.append(new_feat_per_instace)
        new_feat = torch.stack(new_feat)
        assert new_feat.size() == feat1.size()
        return new_feat
    
    def transform_from_two_style(self, cont_img, styl_img1, styl_img2, cont_seg, styl_seg1, styl_seg2, comb_r):
        self.__compute_label_info(cont_seg, styl_seg1)

        s1F4, s1F3, s1F2, s1F1 = self.e4.forward_multiple(styl_img1)
        s2F4, s2F3, s2F2, s2F1 = self.e4.forward_multiple(styl_img2)
        s12F4 = self._feature_interpolation(s1F4, s2F4, comb_r)
        s12F3 = self._feature_interpolation(s1F3, s2F3, comb_r)
        s12F2 = self._feature_interpolation(s1F2, s2F2, comb_r)
        s12F1 = self._feature_interpolation(s1F1, s2F1, comb_r)
        styl_seg = styl_seg1

        cF4, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3 = self.e4(cont_img)
        s1F4 = s1F4.data.squeeze(0)
        s12F4 = s12F4.data.squeeze(0)
        cF4 = cF4.data.squeeze(0)
        csF4 = self.__feature_wct(cF4, s1F4, cont_seg, styl_seg, s12F4)
        Im4 = self.d4(csF4, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3)

        cF3, cpool_idx, cpool1, cpool_idx2, cpool2 = self.e3(Im4)
        s1F3 = s1F3.data.squeeze(0)
        s12F3 = s12F3.data.squeeze(0)
        cF3 = cF3.data.squeeze(0)
        csF3 = self.__feature_wct(cF3, s1F3, cont_seg, styl_seg, s12F3)
        Im3 = self.d3(csF3, cpool_idx, cpool1, cpool_idx2, cpool2)

        cF2, cpool_idx, cpool = self.e2(Im3)
        s1F2 = s1F2.data.squeeze(0)
        s12F2 = s12F2.data.squeeze(0)
        cF2 = cF2.data.squeeze(0)
        csF2 = self.__feature_wct(cF2, s1F2, cont_seg, styl_seg, s12F2)
        Im2 = self.d2(csF2, cpool_idx, cpool)

        cF1 = self.e1(Im2)
        s1F1 = s1F1.data.squeeze(0)
        s12F1 = s12F1.data.squeeze(0)
        cF1 = cF1.data.squeeze(0)
        csF1 = self.__feature_wct(cF1, s1F1, cont_seg, styl_seg, s12F1)
        Im1 = self.d1(csF1)
        return Im1

    def __compute_label_info(self, cont_seg, styl_seg):
        if cont_seg.size == False or styl_seg.size == False:
            return
        max_label = np.max(cont_seg) + 1
        self.label_set = np.unique(cont_seg)
        self.label_indicator = np.zeros(max_label)
        for l in self.label_set:
            # if l==0:
            #   continue
            is_valid = lambda a, b: a > 10 and b > 10 and a / b < 100 and b / a < 100
            o_cont_mask = np.where(cont_seg.reshape(cont_seg.shape[0] * cont_seg.shape[1]) == l)
            o_styl_mask = np.where(styl_seg.reshape(styl_seg.shape[0] * styl_seg.shape[1]) == l)
            self.label_indicator[l] = is_valid(o_cont_mask[0].size, o_styl_mask[0].size)

    def __feature_wct(self, cont_feat, styl_feat, cont_seg, styl_seg, bg_styl_feat=None):
        cont_c, cont_h, cont_w = cont_feat.size(0), cont_feat.size(1), cont_feat.size(2)
        styl_c, styl_h, styl_w = styl_feat.size(0), styl_feat.size(1), styl_feat.size(2)
        cont_feat_view = cont_feat.view(cont_c, -1).clone()
        styl_feat_view = styl_feat.view(styl_c, -1).clone()
        if bg_styl_feat is not None:
            bg_styl_feat_view = bg_styl_feat.view(styl_c, -1).clone()

        if cont_seg.size == False or styl_seg.size == False:
            target_feature = self.__wct_core(cont_feat_view, styl_feat_view)
        else:
            target_feature = cont_feat.view(cont_c, -1).clone()
            if len(cont_seg.shape) == 2:
                t_cont_seg = np.asarray(Image.fromarray(cont_seg).resize((cont_w, cont_h), Image.NEAREST))
            else:
                t_cont_seg = np.asarray(Image.fromarray(cont_seg, mode='RGB').resize((cont_w, cont_h), Image.NEAREST))
            if len(styl_seg.shape) == 2:
                t_styl_seg = np.asarray(Image.fromarray(styl_seg).resize((styl_w, styl_h), Image.NEAREST))
            else:
                t_styl_seg = np.asarray(Image.fromarray(styl_seg, mode='RGB').resize((styl_w, styl_h), Image.NEAREST))

            for l in self.label_set:
                if self.fg_only and l == 0: #background
                    continue
                if self.label_indicator[l] == 0:
                    continue
                cont_mask = np.where(t_cont_seg.reshape(t_cont_seg.shape[0] * t_cont_seg.shape[1]) == l)
                styl_mask = np.where(t_styl_seg.reshape(t_styl_seg.shape[0] * t_styl_seg.shape[1]) == l)
                if cont_mask[0].size <= 0 or styl_mask[0].size <= 0:
                    continue

                cont_indi = torch.LongTensor(cont_mask[0])
                styl_indi = torch.LongTensor(styl_mask[0])
                if self.is_cuda:
                    cont_indi = cont_indi.cuda(self.gpu_id)
                    styl_indi = styl_indi.cuda(self.gpu_id)

                cFFG = torch.index_select(cont_feat_view, 1, cont_indi)
                if (bg_styl_feat is not None) and (l == 0): #background
                    sFFG = torch.index_select(bg_styl_feat_view, 1, styl_indi)
                else:
                    sFFG = torch.index_select(styl_feat_view, 1, styl_indi)    
                tmp_target_feature = self.__wct_core(cFFG, sFFG)
                if torch.__version__ >= "0.4.0":
                    # This seems to be a bug in PyTorch 0.4.0 to me.
                    new_target_feature = torch.transpose(target_feature, 1, 0)
                    new_target_feature.index_copy_(0, cont_indi, \
                            torch.transpose(tmp_target_feature,1,0))
                    target_feature = torch.transpose(new_target_feature, 1, 0)
                else:
                    target_feature.index_copy_(1, cont_indi, tmp_target_feature)

        target_feature = target_feature.view_as(cont_feat)
        ccsF = target_feature.float().unsqueeze(0)
        return ccsF
    
    def __wct_core(self, cont_feat, styl_feat):
        cFSize = cont_feat.size()
        c_mean = torch.mean(cont_feat, 1)  # c x (h x w)
        c_mean = c_mean.unsqueeze(1).expand_as(cont_feat)
        cont_feat = cont_feat - c_mean
        
        iden = torch.eye(cFSize[0])  # .double()
        if self.is_cuda:
            iden = iden.cuda(self.gpu_id)
        #To avoid division by zero
        small_num = 0.00001
        cFdivisor = cFSize[1] - 1.
        if cFdivisor == 0.:
            cFdivisor = small_num
        contentConv = torch.mm(cont_feat, cont_feat.t()).div(cFdivisor) + iden
        # del iden
        try:
            c_u, c_e, c_v = torch.svd(contentConv, some=False)
        except:
            c_e2, c_v = torch.eig(contentConv, True)
            c_e = c_e2[:,0]
        
        k_c = cFSize[0]
        for i in range(cFSize[0] - 1, -1, -1):
            if c_e[i] >= 0.00001:
                k_c = i + 1
                break
        
        sFSize = styl_feat.size()
        s_mean = torch.mean(styl_feat, 1)
        styl_feat = styl_feat - s_mean.unsqueeze(1).expand_as(styl_feat)
        #To avoid division by zero
        small_num = 0.00001
        cFdivisor = cFSize[1] - 1.
        if cFdivisor == 0.:
            cFdivisor = small_num
        styleConv = torch.mm(styl_feat, styl_feat.t()).div(cFdivisor)
        s_u, s_e, s_v = torch.svd(styleConv, some=False)
        
        k_s = sFSize[0]
        for i in range(sFSize[0] - 1, -1, -1):
            if s_e[i] >= 0.00001:
                k_s = i + 1
                break
        
        c_d = (c_e[0:k_c]).pow(-0.5)
        step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
        step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
        whiten_cF = torch.mm(step2, cont_feat)

        s_d = (s_e[0:k_s]).pow(0.5)
        targetFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t())), whiten_cF)
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
        return targetFeature
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, input):
        with torch.no_grad():
            cF4, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3 = self.e4(input)
        Im4 = self.d4(cF4, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3)

        with torch.no_grad():
            cF3, cpool_idx, cpool1, cpool_idx2, cpool2 = self.e3(input)
        Im3 = self.d3(cF3, cpool_idx, cpool1, cpool_idx2, cpool2)

        with torch.no_grad():
            cF2, cpool_idx, cpool = self.e2(input)
        Im2 = self.d2(cF2, cpool_idx, cpool)

        with torch.no_grad():
            cF1 = self.e1(input)
        Im1 = self.d1(cF1)
        return Im1, Im2, Im3, Im4

if __name__ == "__main__":
    model = PhotoWCT()
    Im1, Im2, Im3, Im4 = model(torch.rand(1, 3, 256, 256))
    print(Im1.size(), Im2.size(), Im3.size(), Im4.size())

