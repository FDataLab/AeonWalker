from script.utils.util import logger
from script.models.EvolveGCN.EGCN import EvolveGCN
# from script.models.DynModels import DGCN
from script.models.HTGN import HTGN
# from script.models.static_baselines import VGAENet, GCNNet


def load_model(args):
    if args.model in ['GRUGCN', 'DynGCN']:
        print(args.model)
        # model = DGCN(args)
    elif args.model == 'HTGN':
        print(args.model)
        model = HTGN(args)
    elif args.model == 'EGCN':
        print(args.model)
        model = EvolveGCN(args)
    elif args.model == 'GAE':
        print(args.model)
        # model = GCNNet()
    elif args.model == 'VGAE':
        print(args.model)
        # model = VGAENet()
    else:
        raise Exception('pls define the model')
    logger.info('using model {} '.format(args.model))
    return model
