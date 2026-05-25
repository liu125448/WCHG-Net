import torch 
import torch.optim as optim

def build_optim(net,opt,lr):
    if opt == 'ADM':
        params=[]
        for key, value in net.named_parameters():
            if not value.requires_grad:
                continue
            lr_temp = lr * 0.1
            weight_decay = 1e-4
            if "bias" in key:
                lr_temp = lr_temp * 2
                weight_decay =  0.0
            if "bottleneck" in key:
                lr_temp =lr

            if "classifier" in key:
                lr_temp =lr
              
            params += [{"params": [value], "lr": lr_temp, "weight_decay": weight_decay}]

        optimizer = optim.Adam(params,        
            betas=(0.9, 0.999),  
            eps=1e-3,
        )
        
    elif opt == 'SGD':
        ignored_params = list(map(id, net.bottleneck.parameters())) \
                         + list(map(id, net.classifier.parameters()))\
                       #  + list(map(id, net.gat.parameters())) 
    
        base_params = [p for p in net.parameters() if id(p) not in ignored_params]
    
        deep_optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * lr},
            {'params': net.bottleneck.parameters(), 'lr': lr},
            {'params': net.classifier.parameters(), 'lr': lr}],
            weight_decay=5e-4, momentum=0.9, nesterov=True)

        shallow_optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)
        
        optimizer=deep_optimizer,shallow_optimizer
    
    elif opt == 'SGD_1':
        cmgr_params = list(map(id, net.cmgr.parameters())) if hasattr(net, 'cmgr') else []
        ignored_params = list(map(id, net.bottleneck.parameters())) \
                         + list(map(id, net.classifier.parameters()))\
                         + cmgr_params
                       #  + list(map(id, net.gat.parameters())) 
    
        base_params = [p for p in net.parameters() if id(p) not in ignored_params]
    
        deep_optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * lr},
            {'params': net.bottleneck.parameters(), 'lr': lr},
            {'params': net.cmgr.parameters(), 'lr': 0.5*lr},
            {'params': net.classifier.parameters(), 'lr': lr}],
            weight_decay=5e-4, momentum=0.9, nesterov=True)

        shallow_optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)
        
        optimizer=deep_optimizer,shallow_optimizer
        
    elif opt == 'SGD_2':
        cmgr_params = list(map(id, net.cmgr.parameters())) if hasattr(net, 'cmgr') else []
        ignored_params = list(map(id, net.bottleneck.parameters())) \
                         + list(map(id, net.classifier.parameters()))\
                         + cmgr_params
                       #  + list(map(id, net.gat.parameters())) 
    
        base_params = [p for p in net.parameters() if id(p) not in ignored_params]
    
        deep_optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * lr},
            {'params': net.bottleneck.parameters(), 'lr': lr},
            {'params': net.cmgr.parameters(), 'lr': 0.55*lr},
            {'params': net.classifier.parameters(), 'lr': lr}],
            weight_decay=5e-4, momentum=0.9, nesterov=True)

        shallow_optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)
        
        optimizer=deep_optimizer,shallow_optimizer

    elif opt == 'SGD_3':
        cmgr_params = list(map(id, net.cmgr.parameters())) if hasattr(net, 'cmgr') else []
        ignored_params = list(map(id, net.bottleneck.parameters())) \
                         + list(map(id, net.classifier.parameters()))\
                         + cmgr_params
                       #  + list(map(id, net.gat.parameters())) 
    
        base_params = [p for p in net.parameters() if id(p) not in ignored_params]
    
        deep_optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * lr},
            {'params': net.bottleneck.parameters(), 'lr': lr},
            {'params': net.cmgr.parameters(), 'lr': 0.55*lr},
            {'params': net.classifier.parameters(), 'lr': lr}],
            weight_decay=5e-4, momentum=0.9, nesterov=True)

        shallow_optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)
        
        optimizer=deep_optimizer,shallow_optimizer

    elif opt == 'ADM_ORI':
        ignored_params = list(map(id, net.bottleneck.parameters())) \
                         + list(map(id, net.classifier.parameters()))
    
        base_params = [p for p in net.parameters() if id(p) not in ignored_params]

        optimizer = optim.Adam([{'params': base_params, 'lr': 0.1 * lr,"weight_decay": 0.00004},
            {'params': net.bottleneck.parameters(), 'lr': lr,"weight_decay": 0.0},
            {'params': net.classifier.parameters(), 'lr': lr,"weight_decay": 0.0}],        
                betas=(0.9, 0.999),  
                eps=1e-8,
            )


        
    return optimizer
