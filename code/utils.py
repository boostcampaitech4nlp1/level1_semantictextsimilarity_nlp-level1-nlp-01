# for utilities
def seed_everything(seed:int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def model_selector(name):

    model = ''
    if 'sdlkfjsf':
        continue
    else:
        continue
    
    return model

def lossfct_selector(name):

    loss_fct = ''

    return loss_fct

def optimizer_selector(name):

    Optimizer = ''

    return Optimizer