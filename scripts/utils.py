
import re 
import pandas as pd 
from openpyxl import load_workbook


def filter_list_by_index(list, mask, inverse=False):
    ret = []
    for i, e in enumerate(list):
        if inverse: 
            m = not mask[i]
        else: 
            m = mask[i]
        if m: ret.append(e)
    return ret


def txt_to_set(txt, encoding="utf-8"):
    out = []
    with open(txt, "r", encoding=encoding) as f:
        w = f.readline()
        while w:
            out.append(w.strip('\n'))
            w = f.readline()    
    return set(out)


def load_excel(file, sheet):
    """ загрузка xlsx """
    print("Load excel..")
    wb = load_workbook(file)
    sheet = wb.get_sheet_by_name(sheet)
    data = sheet.values
    cols = next(data)
    df = pd.DataFrame(list(data), columns=cols)
    return df


def read_sql_dump(dump_filename, tr=1e6*5):
    data = []
    with open(dump_filename, 'rb') as f:
        idx_line = 0
        idx_total = 0
        for line in f:
            line = line.decode('utf-8').strip()
            if line.lower().startswith('insert'):
                idx_line += 1
                records = re.findall("\(.*\)", line)
                records = records[0][1:-1].split("),(")
                for i, record in enumerate(records):
                    try:
                        comma_split = record.split(',')
                        record_class = comma_split[-8]
                        record_text = ','.join(comma_split[3:-8]).strip("'")
                        idx_total += 1
                        data.append((record_text, record_class))
                    except:
                        print(idx_line, idx_total, i, records[i-1:i+2])
                        continue
            if idx_total > tr: break
    return data
