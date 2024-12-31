#!/bin/bash

# TODO - run your inference Python3 code
# P1 C set
wget -O ./checkpoint_model/P1/Fine_tune/C/best_val_acc_model.pth 'https://www.dropbox.com/scl/fi/zzqixhrmsg1rmsnnkf5ke/best_val_acc_model.pth?rlkey=3xap0f6vioobj9r9ss7jmdoju&st=46b8waut&dl=1'
# P2 deeplabv3
wget -O ./checkpoint_model/P2/P2_B/deeplabv3/best_val_mIoU_model.pth 'https://www.dropbox.com/scl/fi/vfucida0t8m97ydz6zwz5/best_val_mIoU_model.pth?rlkey=z2y82uimgl95hlw59c9w903et&st=f19pubma&dl=1'
