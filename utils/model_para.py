import sys
def filter_para(model, args, lr, svd=False):
    backbone_named_params = list(filter(lambda np: np[1].requires_grad, model.backbone.named_parameters()))
    classifier_named_params = list(filter(lambda np: np[1].requires_grad, model.classifier.named_parameters()))

    backbone_params = [param for _, param in backbone_named_params]
    backbone_param_names = [name for name, _ in backbone_named_params]

    classifier_params = [param for _, param in classifier_named_params]
    classifier_param_names = [name for name, _ in classifier_named_params]

    if args.opt == 'opt1':
        if svd:
            return {'params': [{'params': backbone_params, 'name': backbone_param_names, 'svd': True, 'thres': args.thres}, {'params': classifier_params, 'name': classifier_param_names, 'svd': False}], 'lr': lr}
        else:
            return [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': lr}]



