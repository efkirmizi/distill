from collections import OrderedDict


def remove_module(state_dict):
    """Safely strips DataParallel and torch.compile prefixes from state_dict keys."""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        # Strip DataParallel prefix if it exists
        if name.startswith('module.'):
            name = name[7:]
        # Strip torch.compile prefix if it exists
        if name.startswith('_orig_mod.'):
            name = name[10:]
            
        new_state_dict[name] = v
    return new_state_dict


def add_module(old_dict):

    new_dict = dict()
    for x,y in old_dict.items():

        t = 'module.' + x
        new_dict[t] = old_dict[x]
    

    return new_dict
