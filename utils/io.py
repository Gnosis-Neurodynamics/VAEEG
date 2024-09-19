import os, mne, re 
from .check import check_type

def load_raw(datafile_list, preload=False):
    """get raw

    Args:
    datafile_list: list of file names or string
    preload: bool. if data would be preloaded. default: False

    Returns:
    final_raw: mne raw

    """
    raw_list = []
    if type(datafile_list) == list:
        if len(datafile_list) == 0:
            raise IOError
        data_types = []
        for data_file in datafile_list:
            if os.path.isdir(data_file):
                dir_base, dir_name = os.path.split(data_file)
                if dir_name == 'EEGData':
                    raise KeyError('compumedics data input path must endswith .sdy which has same pardir with %s'%data_file)
                else:
                    raise KeyError('can`t identify %s.'%data_file)
            else:
                data_type = os.path.splitext(data_file)[1]
                if data_type == '.edf':
                    raw = mne.io.read_raw_edf(data_file, preload=preload, verbose='WARNING')
                elif data_type == '.EEG':
                    raw = mne.io.read_raw_nihon(data_file, preload=preload, verbose='WARNING')
                elif data_type == '.bdf':
                    raw = mne.io.read_raw_bdf(data_file, preload=preload, verbose='WARNING')
                elif data_type == '.vhdr':
                    raw = mne.io.read_raw_brainvision(data_file, verbose=False)
                elif data_type == '.set':
                    raw = mne.io.read_raw_eeglab(data_file, verbose=False)
                elif data_type == '.cnt':
                    raw = mne.io.read_raw_cnt(data_file, verbose=False)   
                else:
                    continue
                data_types.append(data_type)

            raw_list.append(raw)
        if len(raw_list) == 1:
            final_raw = raw_list[0]
            data_type = data_types[0]
        else:
            if len(set(data_types)) > 1:
                raise IOError
            else:
                data_type = data_types[0]
            final_raw = raw_list[0]
            for raw_ind in range(1, len(raw_list)):
                final_raw.append(raw_list[raw_ind])
                
    elif type(datafile_list) == str:
        if os.path.isdir(datafile_list):
            dir_base, dir_name = os.path.split(datafile_list)
            if dir_name == 'EEGData':
                raise KeyError('compumedics data input path must endswith .sdy which has same pardir with %s'%datafile_list)
            else:
                raise KeyError('can`t identify %s.'%datafile_list)
        else:
            data_type = os.path.splitext(datafile_list)[1]
            if data_type == '.edf':
                final_raw = mne.io.read_raw_edf(datafile_list, preload=preload, verbose='WARNING')
            elif data_type == '.EEG':
                final_raw = mne.io.read_raw_nihon(datafile_list, preload=preload)
            elif data_type == '.bdf':
                final_raw = mne.io.read_raw_bdf(datafile_list, preload=preload, verbose='WARNING')
            elif data_type == '.vhdr':
                final_raw = mne.io.read_raw_brainvision(datafile_list, verbose=False)
            elif data_type == '.set':
                final_raw = mne.io.read_raw_eeglab(datafile_list, verbose=False)
            elif data_type == '.cnt':
                final_raw = mne.io.read_raw_cnt(datafile_list, verbose=False)  
            else:
                raise IOError
    else:
        raise IOError
    return final_raw





def get_files(root_dir, extensions=None, recursive=True,
              dot_matching=False, pattern=None):
    """
    Search file paths with given extensions.

    Args:
        root_dir: str
            path of a directory to scan.
        extensions: list, optional.
            If a list, the given file types are included only, for example ['.py', '.pyx'].
            Default None for all kinds of files.
        recursive: bool, default True
            Whether to search subdirectories recursively.
        dot_matching: bool, default False
            Whether to match files start with '.'
        pattern: str or re.Pattern, optional
            If str, file name with the regular pattern will be selected.
            Default None for no filtering.

    Returns:
        a list of paths
    """
    check_type("root_dir", root_dir, [str])
    check_type("recursive", recursive, [bool])
    check_type("dot_matching", dot_matching, [bool])
    check_type("extensions", extensions, [type(None), list])
    check_type("pattern", pattern, [type(None), str, re.Pattern])

    root_dir = get_full_path(root_dir)

    if not os.path.isdir(root_dir):
        raise NotADirectoryError(root_dir)

    if isinstance(extensions, list):
        ext_lower = []
        for ext in extensions:
            if isinstance(ext, str):
                ext_lower.append(ext.lower())
            else:
                raise TypeError("elements of extension must be string (suffixes of files), "
                                "but got %s" % str(extensions))
        extensions = ext_lower

    res = []
    if recursive:
        for root, _, files in os.walk(root_dir):
            for name in files:
                if __file_cond(root, name, extensions, dot_matching, pattern):
                    res.append(os.path.join(root, name))

    else:
        files = os.listdir(root_dir)
        for name in files:
            if __file_cond(root_dir, name, extensions, dot_matching, pattern):
                res.append(os.path.join(root_dir, name))
    return res


def get_full_path(path):
    """
    Get the full path.

    Args:
        path: str
            input path.

    Returns:
        a string of full path
    """
    check_type("path", path, [str])
    return os.path.realpath(os.path.expanduser(os.path.expandvars(path)))




def __file_cond(file_dir, file_name, ext, dot_match, pat):
    check_type("file_dir", file_dir, [str])
    check_type("file_name", file_name, [str])
    check_type("ext", ext, [type(None), list])
    check_type("dot_match", dot_match, [bool])
    check_type("pat", pat, [type(None), str, re.Pattern])

    if dot_match:
        cond_1 = True
    else:
        cond_1 = not file_name.startswith('.')

    f_name = os.path.join(file_dir, file_name)
    cond_2 = os.path.isfile(f_name)

    if ext is None:
        cond_3 = True
    else:
        cond_3 = os.path.splitext(file_name)[1].lower() in ext

    if pat is None:
        cond_4 = True
    else:
        if re.search(pat, file_name):
            cond_4 = True
        else:
            cond_4 = False

    return cond_1 and cond_2 and cond_3 and cond_4
