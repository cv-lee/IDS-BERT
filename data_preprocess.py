import os
import argparse
import math
import re
import pandas
import pandas as pd

from typing import List, Dict, Optional
from urllib import parse


def isnan(value):
    try:
        return math.isnan(value)
    except:
        return False


def preprocess_get_method(data: pandas.core.series.Series) -> pandas.core.series.Series:
    """

        Preprocess a single row (Get Method) of IDS data sheet to divide each feature

        Args:
            data (pandas.core.series.Series) : single row of IDS data sheet

        Retrun:
            data (pandas.core.series.Series) : Divided features listed below.

                'PAYLOAD', 'APP_PROTO', 'SRC_PORT', 'DST_PORT', 'IMPACT', 'RISK', 'JUDGEMENT',
                'Method', 'Method-URL', 'HTTP', 'Host', 'User-Agent', 'Accept', 'Accept-Encoding',
                'Accept-Language','Accept-Charset', 'Content-Type', 'Content-Length',
                'Connection', 'Cookie', 'Upgrade-Insecure-Requests', 'Pragma',
                'Cache-Control', and 'Body'

    """
    payload = data['PAYLOAD']

    if not ('GET' in payload):
        raise ValueError('Invalid Payload ! (GET Method is not Found)')

    data['Method'] = 'GET'
    data['Method-URL'] = parse.unquote(payload.split('GET')[1].split(' HTTP/')[0])

    if 'HTTP/' in payload:
        data['HTTP'] = payload.split('HTTP/')[1].split('..')[0]

    if 'Host: ' in payload:
        data['Host'] = payload.split('Host: ')[1].split('..')[0]

    if 'User-Agent: ' in payload:
        data['User-Agent'] = payload.split('User-Agent: ')[1].split('..')[0]

    if 'Accept: ' in payload:
        data['Accept'] = payload.split('Accept: ')[1].split('..')[0]

    if 'Accept-Encoding: ' in payload:
        data['Accept-Encoding'] = payload.split('Accept-Encoding: ')[1].split('..')[0]

    if 'Accept-Language: ' in payload:
        data['Accept-Language'] = payload.split('Accept-Language: ')[1].split('..')[0]

    if 'Accept-Charset: ' in payload:
        data['Accept-Charset'] = payload.split('Accept-Charset: ')[1].split('..')[0]

    if 'Content-Type: ' in payload:
        data['Content-Type'] = payload.split('Content-Type: ')[1].split('..')[0]

    if 'Content-Length: ' in payload:
        data['Content-Length'] = payload.split('Content-Length: ')[1].split('..')[0]

    if 'Connection: ' in payload:
        data['Connection'] = payload.split('Connection: ')[1].split('..')[0]

    if 'Cookie: ' in payload:
        data['Cookie'] = payload.split('Cookie: ')[1].split('..')[0]

    if 'Upgrade-Insecure-Requests: ' in payload:
        data['Upgrade-Insecure-Requests'] = payload.split('Upgrade-Insecure-Requests: ')[1].split('..')[0]

    if 'Pragma: ' in payload:
        data['Pragma'] = payload.split('Pragma: ')[1].split('..')[0]

    if 'Cache-Control: ' in payload:
        data['Cache-Control'] = payload.split('Cache-Control: ')[1].split('..')[0]

    data['Body'] = ''

    return data


def preprocess_post_method(data: pandas.core.series.Series) -> pandas.core.series.Series:
    """

        Preprocess a single row (Post Method) of IDS data sheet to divide each feature

        Args:
            data (pandas.core.series.Series) : single row of IDS data sheet

        Retrun:
            data (pandas.core.series.Series) : Divided features listed below.

                'PAYLOAD', 'APP_PROTO', 'SRC_PORT', 'DST_PORT', 'IMPACT', 'RISK', 'JUDGEMENT',
                'Method', 'Method-URL', 'HTTP', 'Host', 'User-Agent', 'Accept', 'Accept-Encoding',
                'Accept-Language','Accept-Charset', 'Content-Type', 'Content-Length',
                'Connection', 'Cookie', 'Upgrade-Insecure-Requests', 'Pragma',
                'Cache-Control', and 'Body'

    """
    payload = data['PAYLOAD']

    if not ('POST' in payload):
        raise ValueError('Invalid Payload ! (POST Method is not Found)')

    data['Method'] = 'POST'
    data['Method-URL'] = parse.unquote(payload.split('POST')[1].split(' HTTP/')[0])

    if 'HTTP/' in payload:
        data['HTTP'] = payload.split('HTTP/')[1].split('..')[0]

    if 'Host: ' in payload:
        data['Host'] = payload.split('Host: ')[1].split('..')[0]

    if 'User-Agent: ' in payload:
        data['User-Agent'] = payload.split('User-Agent: ')[1].split('..')[0]

    if 'Accept: ' in payload:
        data['Accept'] = payload.split('Accept: ')[1].split('..')[0]

    if 'Accept-Encoding: ' in payload:
        data['Accept-Encoding'] = payload.split('Accept-Encoding: ')[1].split('..')[0]

    if 'Accept-Language: ' in payload:
        data['Accept-Language'] = payload.split('Accept-Language: ')[1].split('..')[0]

    if 'Accept-Charset: ' in payload:
        data['Accept-Charset'] = payload.split('Accept-Charset: ')[1].split('..')[0]

    if 'Content-Type: ' in payload:
        data['Content-Type'] = payload.split('Content-Type: ')[1].split('..')[0]

    if 'Content-Length: ' in payload:
        data['Content-Length'] = payload.split('Content-Length: ')[1].split('..')[0]

    if 'Connection: ' in payload:
        data['Connection'] = payload.split('Connection: ')[1].split('..')[0]

    if 'Cookie: ' in payload:
        data['Cookie'] = payload.split('Cookie: ')[1].split('..')[0]

    if 'Upgrade-Insecure-Requests: ' in payload:
        data['Upgrade-Insecure-Requests'] = payload.split('Upgrade-Insecure-Requests: ')[1].split('..')[0]

    if 'Pragma: ' in payload:
        data['Pragma'] = payload.split('Pragma: ')[1].split('..')[0]

    if 'Cache-Control: ' in payload:
        data['Cache-Control'] = payload.split('Cache-Control: ')[1].split('..')[0]

    data['Body'] = re.sub('[.]{2,}', '..', payload).split('..')[-1]

    return data


def preprocess(args: argparse.ArgumentParser)-> None:
    '''
        Preprocess dataset with 2 steps
        Step1. Decompose and extract components from full data
        Step2. Combined all of components and convert to text and label

        Args:
            args.data_dir (str) : directory of csv dataset berfore the preprocessing
            args.data_save_dir (str) : directory of csv dataset after the preprocessing
            args.ignored_keywords (str) : keyword list to ignored data preprocessing
    '''
    pd.set_option('mode.chained_assignment', None) # Turn of the warning sign
    
    # Step1. Decompose and extract components from full data

    sheet = pd.read_excel(args.data_dir)[['PAYLOAD',
                                          'APP_PROTO',
                                          'SRC_PORT',
                                          'DST_PORT',
                                          'IMPACT',
                                          'RISK',
                                          'JUDGEMENT']]

    sheet_analysis = {
            'Post': [],
            'Get': [],
            'Ignored-App-Proto': [],
            'Ignored-Keyword': [],
            'Ignored-No-Method': [],
            'Ignored-Error': []
            }
    ignored_keywords = [ik.strip() for ik in args.ignored_keywords.split(',')]
    
    for i in range(len(sheet)):

        try:
            row = sheet.loc[i].copy()
            
            # Protocol Filtering (Exclude if protocol is not http or https)
            if not any(s in row['APP_PROTO'].lower() for s in ['http', 'https']):
                sheet_analysis['Ignored-App-Proto'] += [row]
            
            # Keyword Filtering (Exclude if payload has specific keywords)
            elif any(s in row['PAYLOAD'] for s in ignored_keywords):
                sheet_analysis['Ignored-Keyword'] += [row]
                
            elif 'GET ' in row['PAYLOAD']:
                row = preprocess_get_method(row)
                sheet_analysis['Get'] += [row]

            elif 'POST ' in row['PAYLOAD']:
                row = preprocess_post_method(row)
                sheet_analysis['Post'] += [row]

            # No Method (Exclude if payload has no method such as 'POST' or 'GET')
            else:
                sheet_analysis['Ignored-No-Method'] += [row]

        except:
            # Error logging
            sheet_analysis['Ignored-Error'] += [row]
            pass

    for key in sheet_analysis.keys():
        if len(sheet_analysis[key]) == 0: continue
        sheet_analysis[key] = pd.concat(sheet_analysis[key], axis=1).T
    
    dataset_prep1 = pd.concat([sheet_analysis['Post'], sheet_analysis['Get']], axis=0)
    dataset_prep1_dir = os.path.join(args.data_save_dir, 'dataset_prep1.csv')
    dataset_prep1.to_csv(dataset_prep1_dir, sep=',', index = False, escapechar='\\')

    print(f'<< Filename : {args.data_dir.split("/")[-1].split(".")[0]} >>\n')
    print(f'ㅁ Number of Total data : {len(sheet)}\n')
    print(f'ㅁ Number of GET data : {len(sheet_analysis["Get"])}\n')
    print(f'ㅁ Number of POST data : {len(sheet_analysis["Post"])}\n')
    print(f'ㅁ Number of INGORED data : {len(sheet_analysis["Ignored-App-Proto"])+len(sheet_analysis["Ignored-Keyword"])+len(sheet_analysis["Ignored-No-Method"])+len(sheet_analysis["Ignored-Error"])}')
    print(f'  ㅇ App-Proto : {len(sheet_analysis["Ignored-App-Proto"])}')
    print(f'  ㅇ Keyword : {len(sheet_analysis["Ignored-Keyword"])}')
    print(f'  ㅇ No-Method: {len(sheet_analysis["Ignored-No-Method"])}')
    print(f'  ㅇ Error: {len(sheet_analysis["Ignored-Error"])}\n\n')


    # Step2. Combined all of components and convert to text and label

    sheet = pd.read_csv(dataset_prep1_dir)[['APP_PROTO',
                                            'SRC_PORT',
                                            'DST_PORT',
                                            'IMPACT',
                                            'RISK',
                                            'JUDGEMENT',
                                            'Method',
                                            'Method-URL',
                                            'HTTP',
                                            'Host',
                                            'User-Agent',
                                            'Accept',
                                            'Accept-Encoding',
                                            'Accept-Language',
                                            'Accept-Charset',
                                            'Content-Type',
                                            'Content-Length',
                                            'Connection',
                                            'Cookie',
                                            'Upgrade-Insecure-Requests',
                                            'Pragma',
                                            'Cache-Control',
                                            'Body']]
    sheet['text'] = ''
    sheet['label'] = ''

    for i in range(len(sheet)):
        text = ''
        row = sheet.loc[i].copy()

        if not isnan(row['APP_PROTO']):
            text += f'APP_PROTO: {row["APP_PROTO"]}\n'

        if not isnan(row['SRC_PORT']):
            text += f'SRC_PORT: {row["SRC_PORT"]}\n'

        if not isnan(row['DST_PORT']):
            text += f'DST_PORT: {row["DST_PORT"]}\n'

        if not isnan(row['IMPACT']):
            text += f'IMPACT: {row["IMPACT"]}\n'

        if not isnan(row['RISK']):
            text += f'RISK: {row["RISK"]}\n'

        if not isnan(row['Method']):
            text += f'Method: {row["Method"]}\n'

        if not isnan(row['Method-URL']):
            text += f'Method-URL: {row["Method-URL"]}\n'

        if not isnan(row['HTTP']):
            text += f'HTTP: {row["HTTP"]}\n'

        if not isnan(row['Host']):
            text += f'Host: {row["Host"]}\n'

        if not isnan(row['User-Agent']):
            text += f'User-Agent: {row["User-Agent"]}\n'

        if not isnan(row['Accept']):
            text += f'Accept: {row["Accept"]}\n'

        if not isnan(row['Accept-Encoding']):
            text += f'Accept-Encoding: {row["Accept-Encoding"]}\n'

        if not isnan(row['Accept-Language']):
            text += f'Accept-Language: {row["Accept-Language"]}\n'

        if not isnan(row['Accept-Charset']):
            text += f'Accept-Charset: {row["Accept-Charset"]}\n'

        if not isnan(row['Content-Type']):
            text += f'Content-Type: {row["Content-Type"]}\n'

        if not isnan(row['Content-Length']):
            text += f'Content-Length: {row["Content-Length"]}\n'

        if not isnan(row['Connection']):
            text += f'Connection: {row["Connection"]}\n'

        if not isnan(row['Cookie']):
            text += f'Cookie: {row["Cookie"]}\n'

        if not isnan(row['Upgrade-Insecure-Requests']):
            text += f'Upgrade-Insecure-Requests: {row["Upgrade-Insecure-Requests"]}\n'

        if not isnan(row['Pragma']):
            text += f'Pragma: {row["Pragma"]}\n'

        if not isnan(row['Cache-Control']):
            text += f'Cache-Control: {row["Cache-Control"]}\n'

        if not isnan(row['Body']):
            text += f'Body: {row["Body"]}\n'

        label = int(int(row['JUDGEMENT'])>0)

        sheet['text'][i] = text
        sheet['label'][i] = label
    
    dataset_prep2 = sheet[['text', 'label']]
    dataset_prep2_dir = os.path.join(args.data_save_dir, 'dataset_prep2.csv')
    dataset_prep2.to_csv(dataset_prep2_dir, sep=',', index = False, escapechar='\\')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", "-d", type=str, default="./dataset/dataset_org.xlsx",
                        help="Original dataset directory")

    parser.add_argument("--data_save_dir", "-s", type=str, default="./dataset",
                        help="Preprocessed dataset save directory")

    parser.add_argument("--ignored_keywords", "-i", type=str, default="Qualys, NCSC, BlueCoat",
                        help="If single data include the keyword, it is excluded while preprocessing")

    args = parser.parse_args()

    preprocess(args)