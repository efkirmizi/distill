


def remove_module(old_dict):

    new_dict = dict()
    for x,y in old_dict.items():
        if x.startswith('module'):
            t=x.replace('module.','')
            new_dict[t]=old_dict[x]
        else:
            new_dict[x]=old_dict[x]
    

    return new_dict

def add_module(old_dict):

    new_dict = dict()
    for x,y in old_dict.items():

        t = 'module.' + x
        new_dict[t] = old_dict[x]
    

    return new_dict
