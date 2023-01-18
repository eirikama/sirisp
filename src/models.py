import sys
import torch
from augmented_convolutions import AugmentedConv

class Sirisp(torch.nn.Module):
    def __init__(self):
        super(Sirisp, self).__init__()

        self.conv_1 = AugmentedConv(
            in_channels=1,
            out_channels=32,
            kernel_size=11,
            dk=40,
            dv=4,
            Nh=4,
            stride=1,
            shape=512,
        )
        self.conv_2 = AugmentedConv(
            in_channels=32,
            out_channels=32,
            kernel_size=11,
            dk=40,
            dv=4,
            Nh=4,
            stride=1,
            shape=256,
        )
        self.conv_3 = AugmentedConv(
            in_channels=32,
            out_channels=32,
            kernel_size=11,
            dk=40,
            dv=4,
            Nh=4,
            stride=1,
            shape=128,
        )
        self.conv_4 = AugmentedConv(
            in_channels=32,
            out_channels=7,
            kernel_size=11,
            dk=20,
            dv=2,
            Nh=2,
            stride=1,
            shape=32,
        )

        self.conv_inbetween_in = AugmentedConv(
            in_channels=7,
            out_channels=32,
            kernel_size=11,
            dk=20,
            dv=2,
            Nh=2,
            stride=1,
            shape=32,
        )

        self.conv_5_in = AugmentedConv(
            in_channels=7,
            out_channels=7,
            kernel_size=11,
            dk=20,
            dv=2,
            Nh=2,
            stride=1,
            shape=32,
        )

        self.conv_6_in = AugmentedConv(
            in_channels=32,
            out_channels=32,
            kernel_size=11,
            dk=40,
            dv=4,
            Nh=4,
            stride=1,
            shape=128,
        )

        self.conv_7_in = AugmentedConv(
            in_channels=32,
            out_channels=32,
            kernel_size=11,
            dk=40,
            dv=4,
            Nh=4,
            stride=1,
            shape=256,
        )

        self.conv_8_in = AugmentedConv(
            in_channels=32,
            out_channels=32,
            kernel_size=11,
            dk=40,
            dv=4,
            Nh=4,
            stride=1,
            shape=512,
        )

        self.conv_final_in = torch.nn.Conv1d(
            in_channels=32, out_channels=1, kernel_size=9, padding=4
        )

        self.conv_inbetween_out = AugmentedConv(
            in_channels=7,
            out_channels=32,
            kernel_size=11,
            dk=20,
            dv=2,
            Nh=2,
            stride=1,
            shape=32,
        )

        self.conv_5_out = AugmentedConv(
            in_channels=7,
            out_channels=7,
            kernel_size=11,
            dk=20,
            dv=2,
            Nh=2,
            stride=1,
            shape=32,
        )

        self.conv_6_out = AugmentedConv(
            in_channels=32,
            out_channels=32,
            kernel_size=11,
            dk=40,
            dv=4,
            Nh=4,
            stride=1,
            shape=128,
        )

        self.conv_7_out = AugmentedConv(
            in_channels=32,
            out_channels=32,
            kernel_size=11,
            dk=40,
            dv=4,
            Nh=4,
            stride=1,
            shape=256,
        )

        self.conv_8_out = AugmentedConv(
            in_channels=32,
            out_channels=32,
            kernel_size=11,
            dk=40,
            dv=4,
            Nh=4,
            stride=1,
            shape=512,
        )

        self.conv_final_out = torch.nn.Conv1d(
            in_channels=32, out_channels=1, kernel_size=9, padding=4
        )

        self.bn_1 = torch.nn.BatchNorm1d(32)
        self.bn_2 = torch.nn.BatchNorm1d(32)
        self.bn_3 = torch.nn.BatchNorm1d(32)
        self.bn_4 = torch.nn.BatchNorm1d(7)
        self.bn_5_in = torch.nn.BatchNorm1d(7)
        self.bn_inbetween6_in = torch.nn.BatchNorm1d(32)
        self.bn_6_in = torch.nn.BatchNorm1d(32)
        self.bn_7_in = torch.nn.BatchNorm1d(32)
        self.bn_8_in = torch.nn.BatchNorm1d(32)
        self.bn_5_out = torch.nn.BatchNorm1d(7)
        self.bn_inbetween6_out = torch.nn.BatchNorm1d(32)
        self.bn_6_out = torch.nn.BatchNorm1d(32)
        self.bn_7_out = torch.nn.BatchNorm1d(32)
        self.bn_8_out = torch.nn.BatchNorm1d(32)

        self.maxpool = torch.nn.MaxPool1d(2, return_indices=True)
        self.maxunpool = torch.nn.MaxUnpool1d(2)
        self.leaky = torch.nn.LeakyReLU()
        self.relu = torch.nn.ReLU()

        self.fc1 = torch.nn.Linear(322, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 4)

        self.fcup_in = torch.nn.Linear(322, 322)
        self.fcup_out = torch.nn.Linear(322, 322)
        self.fcup_param = torch.nn.Linear(322, 322)

    def forward(self, x):
        acnn1 = self.leaky(self.conv_1(x))
        acnn1_add, idx_1 = self.maxpool(self.bn_1(acnn1 + x))
        acnn2 = self.leaky(self.conv_2(acnn1_add))
        acnn2_add, idx_2 = self.maxpool(self.bn_2(acnn2 + acnn1_add))
        acnn3 = self.leaky(self.conv_3(acnn2_add))
        acnn3_add, idx_3 = self.maxpool(self.bn_3(acnn3 + acnn2_add))
        acnn4 = self.leaky(self.conv_4(acnn3_add))
        acnn4_add, idx_4 = self.maxpool(self.bn_4(acnn4))

        acnn4_ = acnn4_add.reshape(acnn4_add.size(0), -1)
        up_param = self.leaky(self.fcup_param(acnn4_))

        bn = self.leaky(self.fc1(up_param))

        # Predict Parameters
        out = self.leaky(self.fc2(bn))
        out = self.fc3(out)

        # Inner Shell Correction

        up_in = self.leaky(self.fcup_in(acnn4_))
        up_in = up_in.view(acnn4_add.size())

        inner_upacnn1 = self.leaky(self.conv_5_in(up_in))
        inner_upsampled1 = self.maxunpool(
            self.bn_5_in(inner_upacnn1 + up_in), idx_4, output_size=idx_3.size()
        )
        inner_upsampled1_ = self.bn_inbetween6_in(
            self.leaky(self.conv_inbetween_in(inner_upsampled1))
        )

        inner_upacnn2 = self.leaky(self.conv_6_in(inner_upsampled1_))
        inner_upsampled2 = self.maxunpool(
            self.bn_6_in(inner_upacnn2 + inner_upsampled1_),
            idx_3,
            output_size=idx_2.size(),
        )

        inner_upacnn3 = self.leaky(self.conv_7_in(inner_upsampled2))
        inner_upsampled3 = self.maxunpool(
            self.bn_7_in(inner_upacnn3 + inner_upsampled2),
            idx_2,
            output_size=idx_1.size(),
        )

        inner_upacnn4 = self.leaky(self.conv_8_in(inner_upsampled3))
        inner_upsampled4 = self.maxunpool(
            self.bn_8_in(inner_upacnn4 + inner_upsampled3), idx_1, output_size=x.size()
        )

        inner = self.conv_final_in(inner_upsampled4)

        # Outer Shell Correction

        up_out = self.leaky(self.fcup_in(acnn4_))
        up_out = up_out.view(acnn4_add.size())

        outer_upacnn1 = self.leaky(self.conv_5_out(up_out))
        outer_upsampled1 = self.maxunpool(
            self.bn_5_out(outer_upacnn1 + up_out), idx_4, output_size=idx_3.size()
        )
        outer_upsampled1_ = self.bn_inbetween6_out(
            self.leaky(self.conv_inbetween_out(outer_upsampled1))
        )

        outer_upacnn2 = self.leaky(self.conv_6_out(outer_upsampled1_))
        outer_upsampled2 = self.maxunpool(
            self.bn_6_out(outer_upacnn2 + outer_upsampled1_),
            idx_3,
            output_size=idx_2.size(),
        )

        outer_upacnn3 = self.leaky(self.conv_7_out(outer_upsampled2))
        outer_upsampled3 = self.maxunpool(
            self.bn_7_out(outer_upacnn3 + outer_upsampled2),
            idx_2,
            output_size=idx_1.size(),
        )

        outer_upacnn4 = self.leaky(self.conv_8_out(outer_upsampled3))
        outer_upsampled4 = self.maxunpool(
            self.bn_8_out(outer_upacnn4 + outer_upsampled3), idx_1, output_size=x.size()
        )

        outer = self.conv_final_out(outer_upsampled4)

        return out, inner, outer

