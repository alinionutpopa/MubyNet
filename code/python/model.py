from kaffe.tensorflow import Network

class (Network):
    def setup(self):
        (self.feed('image')
             .conv(3, 3, 64, 1, 1, name='conv1_1')
             .conv(3, 3, 64, 1, 1, name='conv1_2')
             .max_pool(2, 2, 2, 2, name='pool1_stage1')
             .conv(3, 3, 128, 1, 1, name='conv2_1')
             .conv(3, 3, 128, 1, 1, name='conv2_2')
             .max_pool(2, 2, 2, 2, name='pool2_stage1')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .conv(3, 3, 256, 1, 1, name='conv3_4')
             .max_pool(2, 2, 2, 2, name='pool3_stage1')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 256, 1, 1, name='conv4_3_CPM')
             .conv(3, 3, 128, 1, 1, name='conv4_4_CPM')
             .conv(3, 3, 128, 1, 1, name='conv5_1_CPM_L2')
             .conv(3, 3, 128, 1, 1, name='conv5_2_CPM_L2')
             .conv(3, 3, 128, 1, 1, name='conv5_3_CPM_L2')
             .conv(1, 1, 512, 1, 1, name='conv5_4_CPM_L2')
             .conv(1, 1, 19, 1, 1, relu=False, name='conv5_5_CPM_L2'))

        (self.feed('conv5_5_CPM_L2', 
                   'conv4_4_CPM')
             .concat(3, name='concat_stage2_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage2_L2_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage2_L2_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage2_L2_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage2_L2_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage2_L2_H36M')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage2_L2_H36M')
             .conv(1, 1, 19, 1, 1, relu=False, name='Mconv7_stage2_L2_H36M'))

        (self.feed('Mconv7_stage2_L2_H36M', 
                   'conv4_4_CPM')
             .concat(3, name='concat_stage3_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage3_L2_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage3_L2_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage3_L2_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage3_L2_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage3_L2_H36M')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage3_L2_H36M')
             .conv(1, 1, 19, 1, 1, relu=False, name='Mconv7_stage3_L2_H36M'))

        (self.feed('Mconv7_stage3_L2_H36M', 
                   'conv4_4_CPM')
             .concat(3, name='concat_stage4_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage4_L2_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage4_L2_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage4_L2_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage4_L2_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage4_L2_H36M')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage4_L2_H36M')
             .conv(1, 1, 19, 1, 1, relu=False, name='Mconv7_stage4_L2_H36M'))

        (self.feed('Mconv7_stage4_L2_H36M', 
                   'conv4_4_CPM')
             .concat(3, name='concat_stage5_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage5_L2_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage5_L2_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage5_L2_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage5_L2_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage5_L2_H36M')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage5_L2_H36M')
             .conv(1, 1, 19, 1, 1, relu=False, name='Mconv7_stage5_L2_H36M'))

        (self.feed('Mconv7_stage5_L2_H36M', 
                   'conv4_4_CPM')
             .concat(3, name='concat_stage6_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage6_L2_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage6_L2_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage6_L2_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage6_L2_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage6_L2_H36M')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage6_L2_H36M')
             .conv(1, 1, 19, 1, 1, relu=False, name='Mconv7_stage6_L2_H36M'))

        (self.feed('Mconv7_stage6_L2_H36M', 
                   'conv4_4_CPM')
             .concat(3, name='concat_stage1_H36M_3D')
             .conv(3, 3, 128, 1, 1, name='conv5_1_CPM_L3')
             .conv(3, 3, 128, 1, 1, name='conv5_2_CPM_L3')
             .conv(3, 3, 128, 1, 1, name='conv5_3_CPM_L3')
             .conv(1, 1, 512, 1, 1, name='conv5_4_CPM_L3')
             .conv(1, 1, 48, 1, 1, relu=False, name='conv5_5_CPM_L3_H36M'))

        (self.feed('Mconv7_stage6_L2_H36M', 
                   'conv5_5_CPM_L3_H36M', 
                   'conv4_4_CPM')
             .concat(3, name='concat_stage2_H36M_3D')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage2_L3_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage2_L3_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage2_L3_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage2_L3_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage2_L3_H36M')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage2_L3_H36M')
             .conv(1, 1, 48, 1, 1, relu=False, name='Mconv7_stage2_L3_H36M'))

        (self.feed('Mconv7_stage6_L2_H36M', 
                   'Mconv7_stage2_L3_H36M', 
                   'conv4_4_CPM')
             .concat(3, name='concat_stage3_H36M_3D')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage3_L3_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage3_L3_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage3_L3_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage3_L3_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage3_L3_H36M')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage3_L3_H36M')
             .conv(1, 1, 48, 1, 1, relu=False, name='Mconv7_stage3_L3_H36M'))

        (self.feed('Mconv7_stage6_L2_H36M', 
                   'Mconv7_stage3_L3_H36M', 
                   'conv4_4_CPM')
             .concat(3, name='concat_stage4_H36M_3D')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage4_L3_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage4_L3_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage4_L3_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage4_L3_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage4_L3_H36M')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage4_L3_H36M')
             .conv(1, 1, 48, 1, 1, relu=False, name='Mconv7_stage4_L3_H36M'))

        (self.feed('Mconv7_stage6_L2_H36M', 
                   'Mconv7_stage4_L3_H36M', 
                   'conv4_4_CPM')
             .concat(3, name='concat_stage5_H36M_3D')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage5_L3_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage5_L3_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage5_L3_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage5_L3_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage5_L3_H36M')
             .conv(1, 1, 128, 1, 1, relu=False, name='Mconv6_stage5_L3_H36M')
             .conv(1, 1, 48, 1, 1, relu=False, name='Mconv7_stage5_L3_H36M'))

        (self.feed('Mconv7_stage6_L2_H36M', 
                   'Mconv7_stage5_L3_H36M', 
                   'conv4_4_CPM')
             .concat(3, name='concat_stage6_H36M_3D')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage6_L3_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage6_L3_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage6_L3_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage6_L3_H36M')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage6_L3_H36M')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage6_L3_H36M')
             .conv(1, 1, 48, 1, 1, relu=False, name='Mconv7_stage6_L3_H36M'))

        (self.feed('Mconv7_stage6_L3_H36M', 
                   'Mconv7_stage5_L3_H36M', 
                   'Mconv7_stage4_L3_H36M', 
                   'Mconv7_stage3_L3_H36M', 
                   'Mconv7_stage2_L3_H36M', 
                   'conv5_5_CPM_L3_H36M')
             .add(name='sum_stage6'))