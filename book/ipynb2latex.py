import codecs
import json
import re
import yaml
import os
from tqdm.notebook import tqdm
from PIL import Image
from base64 import b64decode
from io import BytesIO
import numpy as np
import pykakasi

class AutoIndexing():
    def __init__(self,):
        self.kks = pykakasi.kakasi()
    
    def japanese_check(self, s):
        # s: string
        if re.search(r'[ぁ-ん]+|[ァ-ヴー]+|[一-龠]+', s):
            return True
        else:
            return False
        
    def converter(self, s):
        s = s.replace(r"**", "")
        if self.japanese_check(s):
            s_hira = "".join([word["hira"] for word in self.kks.convert(s)])
            return r'\textbf{'+s+r'}\index{'+s_hira + r"@" + s + r'}'
            #return r'\index{'+s_hira + r"@\textbf{" + s + r'}|textbf}'
        else:
            return r'\textbf{'+s+r'}\index{'+s + r'}'
        
def markdown2latex(s, auto_indexing=AutoIndexing()):
    # s: string
    s = re.sub(r'\####\ (.+?)\n', r'\\paragraph{\1}\n', s)  # subsubsection
    s = re.sub(r'\###\ (.+?)\n', r'\\subsubsection{\1}\n', s)  # subsubsection
    s = re.sub(r'\##\ (.+?)\n', r'\\subsection{\1}\n', s)  # subsection
    s = re.sub(r'\#\ (.+?)\n', r'\\section{\1}\n', s)      # section
    
    #s = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\\index{\1}}', s)   # bold
    s = re.sub(r'\*\*(.+?)\*\*', lambda m: auto_indexing.converter(m.group()), s)   # bold
    #s = re.sub(r'\*(.+?)\*', r'\\textit{\1}', s)       # italic
    
    s = s.replace(r"```{note}", "\\footnote{") # note to footnote
    s = s.replace(r"```", "}")
    s = re.sub(r'<(.+?)>', r'\\url{\1}', s) # url

    s = re.sub(r'{cite:p}`(.+?)`', r'\\cite{\1}', s)     
    s = re.sub(r'`(.+?)`', r'\\jl{\1}', s) # inline code with \newcommand{\jl}{\lstinline[language=julia]}

    s = re.sub(r'<(.+?)>', r'\\url{\1}', s) # url
    s = s.replace(r":=", "\\triangleq")
    s = s.replace(r"（", " (") 
    s = s.replace(r"）", ") ") 
    s = s.replace(r"$$", "") 
    #s = s.replace(r"．", "．\n") 
    return s

def latex_itemized(text):
    #splited_text = text.split('\n')
    #splited_text = all_remove(splited_text, "\n")
    splited_text = list(filter(None, text))
    # itemize
    item_idx = [line[:2] == "- " for line in splited_text]
    if np.sum(item_idx) > 0:
        item_idx += [False]
        item_startend = np.where(np.diff(np.array(item_idx)) == True)[0]
        item_startend += np.arange(len(item_startend)) + 1

        # replace - to \item
        for i in range(len(splited_text)):
            if item_idx[i]:
                splited_text[i] = splited_text[i].replace('- ', '\item ', 1) 

        # add begin and end
        for j in range(len(item_startend)):
            if j % 2 == 0:
                splited_text.insert(item_startend[j], "\\begin{itemize}")
            else:
                splited_text.insert(item_startend[j], "\\end{itemize}")
    
    # enumerate
    enum_idx = [line[:3] == "1. " for line in splited_text]
    if np.sum(enum_idx) > 0:
        enum_idx += [False]
        enum_startend = np.where(np.diff(np.array(enum_idx)) == True)[0]
        enum_startend += np.arange(len(enum_startend)) + 1

        # replace 1. to \item
        for i in range(len(splited_text)):
            if enum_idx[i]:
                splited_text[i] = splited_text[i].replace('1. ', '\item ', 1) 

        # add begin and end
        for j in range(len(enum_startend)):
            if j % 2 == 0:
                splited_text.insert(enum_startend[j], "\\begin{enumerate}")
            else:
                splited_text.insert(enum_startend[j], "\\end{enumerate}")

    for i in range(len(splited_text)):
        if splited_text[i][-1:] != "\n":
            splited_text[i] += "\n"
    return splited_text

def all_remove(xlist, remove):
    return [value for value in xlist if value != remove]

def md_ipynb2latex(dir_path, filename, auto_indexing=AutoIndexing()):
    save_dir = "./text/" + filename
    os.makedirs(save_dir, exist_ok=True)
    file_path = dir_path + filename
    master_list = []
    if os.path.isfile(file_path + ".md"):
        f = codecs.open(file_path + ".md", 'r', encoding="utf8")
        md = f.read()
        # convert
        text = markdown2latex(md, auto_indexing)
        text = text.split('\n')
        text = latex_itemized(text) #all_remove(text, "\n")
        if not ":filter: docname in docnames" in "".join(text):
            # save
            master_list += text
    elif os.path.isfile(file_path + ".ipynb"):
        f = codecs.open(file_path + ".ipynb", 'r', encoding="utf8")
        source = f.read()
        y = json.loads(source)
        num_cells = len(y['cells'])
        for cell_idx in range(num_cells):
            cell = y['cells'][cell_idx]
            if cell['cell_type'] == 'markdown':
                # convert
                text = [markdown2latex(s, auto_indexing) for s in cell['source']]
                text = latex_itemized(text)
                if not ":filter: docname in docnames" in "".join(text):
                    # save
                    master_list += text
                    #parted_file_path = save_dir + "/{:03d}.tex".format(cell_idx)
                    #with open(parted_file_path, 'w', encoding='UTF-8') as f:
                    #    f.writelines(text)
                    #master_list.append(r"\input{"+parted_file_path+"}\n")
            elif cell['cell_type'] == 'code':
                # ToDo:'outputs'
                code = cell['source']
                parted_file_path = save_dir + "/{:03d}.jl".format(cell_idx)
                with open(parted_file_path, 'w', encoding='UTF-8') as f:
                    f.writelines(code)
                master_list.append(r"\lstinputlisting[language=julia]{"+parted_file_path+"}\n")

                if cell['outputs']:
                    if 'data' in cell['outputs'][0]:
                        output = cell['outputs'][0]['data']
                        if "image/png" in output.keys():
                            png_bytes = output['image/png']
                            png_bytes = b64decode(png_bytes)
                            bytes_io = BytesIO(png_bytes)
                            image = Image.open(bytes_io)

                            figname = "cell{:03d}.png".format(cell_idx)
                            figsavepath = "./fig/" + filename + "/" + figname
                            os.makedirs("./fig/" + filename, exist_ok=True)
                            image.save(figsavepath, 'png')

                            caption = figname
                            figlabel = figname #"ccc"
                            figcode = "\\begin{figure}[ht]\n\t\centering\n"
                            figcode += "\t\includegraphics[scale=0.8, max width=\linewidth]{"+figsavepath+"}\n"
                            figcode += "\t\caption{" + caption + "}\n"
                            figcode += "\t\label{"+figlabel+"}\n"
                            figcode += "\end{figure}"

                            parted_output_path = save_dir + "/output_{:03d}.tex".format(cell_idx)
                            with open(parted_output_path, 'w', encoding='UTF-8') as f:
                                f.writelines(figcode)
                            master_list.append(r"\input{"+parted_output_path+"}\n")
                        """
                        elif "text/plain" in output.keys():
                            print(output["text/plain"])
                        """

    with open(save_dir + '.tex', 'w', encoding='UTF-8') as f:
        f.writelines(master_list)
    return master_list

if __name__ == "__main__": 
    dir_path = "../contents/"
    with open(dir_path + "_toc.yml") as file:
        toc_yaml = yaml.safe_load(file)
    main_list = []
    auto_indexing = AutoIndexing()
    for i, section in tqdm(enumerate(toc_yaml['sections'])):
        print(section['file']) # intro
        if i > 0:
            for subsection in section['sections']:
                filename = subsection['file']
                print(filename)
                md_ipynb2latex(dir_path, filename, auto_indexing)
                main_list.append(r"\input{./text/"+filename+".tex}\n")
    