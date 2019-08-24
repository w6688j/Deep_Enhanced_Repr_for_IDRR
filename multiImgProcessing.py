from multiprocessing import cpu_count, Manager, Process

import numpy as np

from spacyTool import SpacyTool


def multipro_Text2Img(sentence_list, text_pkl, img_pkl):
    tensors = []
    for sentence in sentence_list:
        try:
            if text_pkl is not None and sentence in set(text_pkl):
                index = text_pkl.index(sentence)
                tensors.append(img_pkl[index] * 100)
            else:
                tensors.append(np.zeros((3, 256, 256)))
        except Exception as e:
            tensors.append(np.zeros((3, 256, 256)))
            print(e)
            continue

    return tensors


def get_phrase_imgs(phrase_img1s, arg1_sen, phrase_text_pkl, phrase_img_pkl, max_phrase_len=3, spacy=None):
    if phrase_img1s is None:
        phrase_img1s = []
    else:
        phrase_img1s = phrase_img1s

    phrase_img_tensor = np.zeros((max_phrase_len, 3 * 256 * 256))
    if len(arg1_sen) == 0:
        phrase_img1s.append(phrase_img_tensor)
    else:
        for arg1_item in arg1_sen:
            arg1_item_phrase = SpacyTool.get_phrase(spacy, arg1_item)
            if arg1_item_phrase is None:
                phrase_img1s.append(phrase_img_tensor)
            else:
                if len(arg1_item_phrase) > max_phrase_len:
                    arg1_item_phrase = arg1_item_phrase[:max_phrase_len]
                phrase_img = multipro_Text2Img(arg1_item_phrase, phrase_text_pkl, phrase_img_pkl)
                length = len(phrase_img)
                if phrase_img is not None:
                    phrase_img = np.array(phrase_img)
                    phrase_img = phrase_img.reshape(len(arg1_item_phrase), 3 * 256 * 256)
                    if length > max_phrase_len:
                        phrase_img_tensor = phrase_img[:max_phrase_len]
                    else:
                        phrase_img_tensor[:length] = phrase_img

                    phrase_img1s.append(phrase_img_tensor)
                else:
                    phrase_img1s.append(phrase_img_tensor)
    return phrase_img1s


def multipro_phrase_imgs(arg1_sen, phrase_text_pkl, phrase_img_pkl, max_phrase_len, spacy):
    rs_list = []
    processes_num = cpu_count()
    length = len(arg1_sen)
    if length < processes_num:
        rs = get_phrase_imgs(rs_list, arg1_sen, phrase_text_pkl, phrase_img_pkl, max_phrase_len, spacy)
        return rs
    else:
        step = length / processes_num
        with Manager() as manager:
            tensors = manager.list()  # <-- can be shared between processes.
            # pool = Pool(processes=cpu_count())
            processes = []
            for i in range(processes_num):
                if (i + 1) * step > length:
                    sentence_list = arg1_sen[int(i * step):]
                else:
                    sentence_list = arg1_sen[int(i * step):int((i + 1) * step)]
                p = Process(target=get_phrase_imgs,
                            args=(tensors, sentence_list, phrase_text_pkl, phrase_img_pkl,
                                  max_phrase_len, spacy))  # Passing the list
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            for p in processes:
                if p.is_alive:
                    p.terminate()
            for item in tensors:
                rs_list.append(list(item))

        return np.array(rs_list)


def multipro_phrase_imgs_mult(return_list, arg1_sen, phrase_text_pkl, phrase_img_pkl, max_phrase_len, spacy):
    rs_list = []
    processes_num = cpu_count()
    length = len(arg1_sen)
    if length < processes_num:
        rs = get_phrase_imgs(rs_list, arg1_sen, phrase_text_pkl, phrase_img_pkl, max_phrase_len, spacy)
        return rs
    else:
        step = length / processes_num
        with Manager() as manager:
            tensors = manager.list()  # <-- can be shared between processes.
            # pool = Pool(processes=cpu_count())
            processes = []
            for i in range(processes_num):
                if (i + 1) * step > length:
                    sentence_list = arg1_sen[int(i * step):]
                else:
                    sentence_list = arg1_sen[int(i * step):int((i + 1) * step)]
                p = Process(target=get_phrase_imgs,
                            args=(tensors, sentence_list, phrase_text_pkl, phrase_img_pkl,
                                  max_phrase_len, spacy))  # Passing the list
                # pool.apply_async(get_phrase_imgs,
                #                  args=(tensors, sentence_list, phrase_text_pkl, phrase_img_pkl))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            for p in processes:
                if p.is_alive:
                    p.terminate()
            # pool.close()
            # pool.join()
            for item in tensors:
                rs_list.append(list(item))

        return_list.append(rs_list)
        # return_list = rs_list
        return return_list
        # return np.array(rs_list)

def multipro_SentText2Img(tensors, sentence_list, text_pkl, img_pkl):
    for sentence in sentence_list:
        try:
            if text_pkl is not None and sentence in set(text_pkl):
                index = text_pkl.index(sentence)
                tensors.append(img_pkl[index])
            else:
                tensors.append(np.zeros((3, 256, 256)))
        except Exception as e:
            tensors.append(np.zeros((3, 256, 256)))
            print(e)
            continue

    return tensors


def multipro_sentence_imgs(arg1_sen, phrase_text_pkl, phrase_img_pkl):
    rs_list = []
    processes_num = cpu_count()
    length = len(arg1_sen)
    if length < processes_num:
        rs = multipro_SentText2Img(rs_list, arg1_sen, phrase_text_pkl, phrase_img_pkl)
        return rs
    else:
        step = length / processes_num
        with Manager() as manager:
            tensors = manager.list()  # <-- can be shared between processes.
            processes = []
            for i in range(processes_num):
                if (i + 1) * step > length:
                    sentence_list = arg1_sen[int(i * step):]
                else:
                    sentence_list = arg1_sen[int(i * step):int((i + 1) * step)]
                p = Process(target=multipro_SentText2Img,
                            args=(tensors, sentence_list, phrase_text_pkl, phrase_img_pkl))  # Passing the list
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            for p in processes:
                if p.is_alive:
                    p.terminate()
            for item in tensors:
                rs_list.append(list(item))

        return np.array(rs_list)


def multipro_phrase_imgs_all(arg1_sen, arg2_sen, phrase_text_pkl,
                             phrase_img_pkl, max_phrase_len, spacy):
    with Manager() as manager:
        return_list = manager.list()
        processes = []
        for i in range(2):
            if i == 0:
                sentence_list = arg1_sen
            else:
                sentence_list = arg2_sen
            p = Process(target=multipro_phrase_imgs_mult,
                        args=(return_list,
                              sentence_list, phrase_text_pkl, phrase_img_pkl, max_phrase_len,
                              spacy))  # Passing the list
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        for p in processes:
            if not p.is_alive:
                p.terminate()

        return np.array(return_list[0]), np.array(return_list[1])
