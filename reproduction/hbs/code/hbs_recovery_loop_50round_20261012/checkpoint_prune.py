"""Independent flat checkpoint-and-prune retention comparator."""
import hashlib
import shutil

def retain_flat_objects(root, keep):
    destinations = {}
    for name, retained in keep.items():
        old = root/'objects'/name
        if retained:
            destinations[name] = 'objects/'+name
            continue
        target = root/'archive'/name
        shutil.copyfile(old,target)
        if hashlib.sha256(old.read_bytes()).digest()!=hashlib.sha256(target.read_bytes()).digest():
            raise ValueError('Archive copy verification failed; original retained')
        old.unlink()
        destinations[name] = 'archive/'+name
    return destinations
