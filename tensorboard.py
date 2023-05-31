from torch.utils.tensorboard import SummaryWriter
import torch
writer = SummaryWriter("ZCH_Tensorboard_Trying_logs")      #第一个参数指明 writer 把summary内容 写在哪个目录下
baseline_data = torch.load('./datas/baseline.pth')
aug = torch.load('./datas/baseline_aug_data.pth')
baseline_cutoutonly=torch.load('./datas/cutout_only.pth')
baseline_cutout=torch.load('./datas/cutout_aug.pth')
baseline_mixuponly=torch.load('./datas/mixup_only.pth')
baseline_mixup=torch.load('./datas/mixup_aug.pth')
cutmixonly=torch.load('./datas/cutmix_only.pth')
cutmix=torch.load('./datas/cutmix_aug.pth')
for i in range(len(baseline_data['loss'])):
    writer.add_scalar("Baseline/loss",baseline_data['loss'][i],i)
    writer.add_scalars("Baseline/acc",{'Valid Accuracy':baseline_data['valid_acc'][i],'Test Accuracy':baseline_data['test_acc'][i]},i)
for i in range(100): 
    writer.add_scalars("Aug/loss",{'Without Data Augmentation':baseline_data['loss'][i],'With Data Augmentation':aug['loss'][i]},i)
    writer.add_scalars("Aug/acc",{'Without Data Augmentation':baseline_data['test_acc'][i],'With Data Augmentation':aug['test_acc'][i]},i)
    
    writer.add_scalars("Cutout/loss",{'Baseline':baseline_data['loss'][i],'Cutout':baseline_cutoutonly['loss'][i],'Augmentation':aug['loss'][i],'Augmentation+Cutout':baseline_cutout['loss'][i]},i)
    writer.add_scalars("Cutout/acc",{'Baseline':baseline_data['test_acc'][i],'Cutout':baseline_cutoutonly['test_acc'][i],'Augmentation':aug['test_acc'][i],'Augmentation+Cutout':baseline_cutout['test_acc'][i]},i)

    writer.add_scalars("Mixup/loss",{'Baseline':baseline_data['loss'][i],'Mixup':baseline_mixuponly['loss'][i],'Augmentation':aug['loss'][i],'Augmentation+Mixup':baseline_mixup['loss'][i]},i)
    writer.add_scalars("Mixup/acc",{'Baseline':baseline_data['test_acc'][i],'Mixup':baseline_mixuponly['test_acc'][i],'Augmentation':aug['test_acc'][i],'Augmentation+Mixup':baseline_mixup['test_acc'][i]},i)

    writer.add_scalars("Cutmix/loss",{'Baseline':baseline_data['loss'][i],'Cutmix':cutmixonly['loss'][i],'Augmentation':aug['loss'][i],'Augmentation+Cutmix':cutmix['loss'][i]},i)
    writer.add_scalars("Cutmix/acc",{'Baseline':baseline_data['test_acc'][i],'Cutmix':cutmixonly['test_acc'][i],'Augmentation':aug['test_acc'][i],'Augmentation+Cutmix':cutmix['test_acc'][i]},i)
    #没有augmentation时四种模型的比较
    writer.add_scalars("Without Augmentation/loss",{'Baseline':baseline_data['loss'][i],'Cutout':baseline_cutoutonly['loss'][i],'Mixup':baseline_mixuponly['loss'][i],'Cutmix':cutmixonly['loss'][i]},i)
    writer.add_scalars("Without Augmentation/acc",{'Baseline':baseline_data['test_acc'][i],'Cutout':baseline_cutoutonly['test_acc'][i],'Mixup':baseline_mixuponly['test_acc'][i],'Cutmix':cutmixonly['test_acc'][i]},i)

    #有augmentation时四种模型的比较 
    writer.add_scalars("With Augmentation/loss",{'Baseline':baseline_data['loss'][i],'Cutout':baseline_cutout['loss'][i],'Mixup':baseline_mixup['loss'][i],'Cutmix':cutmix['loss'][i]},i)
    writer.add_scalars("With Augmentation/acc",{'Baseline':baseline_data['test_acc'][i],'Cutout':baseline_cutout['test_acc'][i],'Mixup':baseline_mixup['test_acc'][i],'Cutmix':cutmix['test_acc'][i]},i)
writer.close() 
