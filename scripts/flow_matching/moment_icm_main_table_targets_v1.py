"""Read the72 main-table aggregate targets without inventing branch mappings."""
import re

from scripts.flow_matching.audit_moment_icm_primary_scope_v1 import DATASETS


def decimal_lines(text):
    return [float(v) for v in re.findall(r'(?m)^(\d+\.\d{3})$',text)]


def targets(paper):
    import fitz
    with fitz.open(paper) as document:
        page3,page4=document[2].get_text(),document[3].get_text()
    a=decimal_lines(page3.split('Model / Example')[1].split('Table 1:')[0])
    b=decimal_lines(page4.split('Model name')[1].split('Table 2:')[0])
    third=page4.split('MOMENT-Tiny\n+ ICM')[1].split('Table 3:')[0]
    c=decimal_lines(third)
    wins=re.findall(r'(?m)^(\d+)/(\d+)/(\d+)$',third)
    if (len(a),len(b),len(c),len(wins))!=(42,18,6,2):
        raise ValueError('All main-table source coordinates required')
    result=[]
    for index,model in enumerate(['N-BEATS','MOMENT-Tiny','UP2ME graph adapter','Crossformer','iTransformer','MOIRAI concatenation','ICM']):
        for j,dataset in enumerate(DATASETS):
            result.append(dict(table=1,pdf_page=3,model=model,dataset=dataset,metric='mean_horizon_MSE',reported=a[6*index+j]))
    for index,model in enumerate(['MOMENT-Tiny pretrained','ICM pretrained beta frozen','ICM pretrained beta finetuned']):
        for j,dataset in enumerate(DATASETS):
            result.append(dict(table=2,pdf_page=4,model=model,dataset=dataset,metric='mean_horizon_MSE',reported=b[6*index+j]))
    for index,model in enumerate(['MOMENT-Tiny','MOMENT-Tiny + ICM']):
        for j,metric in enumerate(['mean_dataset_accuracy','std_dataset_accuracy','median_dataset_accuracy']):
            result.append(dict(table=3,pdf_page=4,model=model,dataset='26_multivariate_datasets',metric=metric,reported=c[2*j+index]))
        for metric,value in zip(['dataset_wins','dataset_ties','dataset_losses'],wins[index]):
            result.append(dict(table=3,pdf_page=4,model=model,dataset='26_multivariate_datasets',metric=metric,reported=int(value)))
    if len(result)!=72:raise ValueError('All72 primary aggregate cells required')
    return result
