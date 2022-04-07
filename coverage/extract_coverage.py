#!/usr/bin/python3
import os
#from runpy import run_path
import sys
import argparse
import re
import yaml
from bs4 import BeautifulSoup as bs


# glob_dict={"covergroup": [ 
#                             {
#                                 "name":"", 
#                                 "coverpoints":[{
#                                     "name": "",
#                                     "percentage":"",
#                                     "cross":0/1,
#                                     "<bin_name0>":"hit/miss"
#                                     "<bin_name1>":"hit/miss"
#                                     .
#                                     .
#                                 }]
#                             }
#                          ]

def get_cp_table_dict(table):
    """
    Function to create a coverpoint dictionary from table for Bin distribution
    Input Argument: Beautiful Soup object of table
    Return Value: a dictionary of key = bin name; value = bin hit or miss 1/0
    """
    row_head=[]
    row_dict={}
    table_dict={}
    rows=table.find_all('tr')
    for i,row in enumerate(rows):
        cols=row.find_all('td')
        if i==0:
            for col in cols:
                row_head.append(str(col.string))
        else:
            for j, col in enumerate(cols):
                row_dict[row_head[j]]=str(col.string)
        if row_dict != {}:
            table_dict[row_dict['NAME']]=1 if int(row_dict['COUNT']) > 0 else 0   
        row_dict={}
    return table_dict

def get_cross_table_dict(table):
    """
    Function to Extract the dictionary for Cross coverage
    Input Argument: Beautiful Soup object of table
    Return Value: a dictionary of key = bin name; value = bin hit or miss 1/0
    """
    row_head=[]
    table_dict={}
    bin_name=""
    bin_count=0
    cross_name=""
    rows=table.find_all('tr')
    for i,row in enumerate(rows):
        cols=row.find_all('td')
        if i==0:
            for col in cols:
                row_head.append(str(col.string))

            for head in row_head:
                if re.search(r"cp_*", head):
                    if cross_name=="":
                        cross_name=head
                    else:
                        cross_name+="X"+head
        else:
            for j, col in enumerate(cols):
                col_val=" ".join(col.string.replace("\n","").strip().split())
                if j<len(row_head):
                    if re.search("cp_*",row_head[j]):
                        if bin_name=="":
                            bin_name=col_val
                        else:
                            bin_name+="X"+col_val
                    if re.search("COUNT", row_head[j]) and col_val!='--':
                        bin_count=str(col.string)

        if bin_name!="":
            table_dict[bin_name]=1 if int(bin_count) > 0 else 0
            bin_name=""
    return table_dict

def extract_cp_details(div):
    """
    function to reaturn a list of coverpoint dictionaries
    Input Arg: Beautifl soup object for the div containing the coverpoint and cross tables
    Return Value: List of dictionaries per coverpoint/cross
    """
    
    spans=div.find_all('span', {'class':'repname'})
    all_table=div.find_all('table')
    table_count=0
    row_head=[]
    row_dict={}
    cp_dict={}
    cp_list=[]
    cross_extract=0
    for i,heading in enumerate(spans):
        table_description=" ".join(heading.string.replace("\n","").strip().split())

        if re.search("Summary for Cross", table_description):
            cross_extract=1
            temp=re.search("Summary for Cross (\w+)", table_description)
            current_cp=str(temp.group(1))
            if cp_dict!={}:
                cp_list.append(cp_dict)
                cp_dict={}
            cp_dict['name']=current_cp

        if re.search("Summary for Variable (\w+)", table_description):
            cross_extract=0
            temp=re.search("Summary for Variable (\w+)", table_description)
            current_cp=str(temp.group(1))
            if cp_dict!={}:
                cp_list.append(cp_dict)
                cp_dict={}
            cp_dict['name']=current_cp #Define cp_dict

        if re.search(r"[bB][iI][nN][sS]$", table_description) and not re.search(r"[iI]llegal", table_description):

            if cross_extract==0:
                cp_dict.update(get_cp_table_dict(all_table[table_count]))
            else:
                cp_dict.update(get_cross_table_dict(all_table[table_count]))
        
        if not re.search(r"Automatically Generated", table_description) and not re.search(r"User Defined", table_description):
            table_count+=1            

    return cp_list

def extract_summary(div):
    """
    Function to Extract summary of the covergroup
    and coverage percentage per coverpoint and cross
    Input Args: Beautiful Soup object of the div containing the Summary tables
    Return val: dictionary summarizing the coverpoint and cross results
    """
    head_list=[]
    row_dict={}
    table_dict_list=[]
    table_list=[]
    summary_grp=div.find_all('table')

    for table in summary_grp:
        rows=table.find_all('tr')
        for i, row in enumerate(rows):
            cols = row.find_all('td')
            if i==0:
                for col in cols:
                    head_list.append(col.string)
            else:
                for j, col in enumerate(cols):
                    row_dict[str(head_list[j])]=str(col.string)
            table_dict_list.append(row_dict)
            row_dict={}
        table_list.append(table_dict_list)
        table_dict_list=[]
        head_list=[]

    return table_list

def get_cg_dict_list(c_grp):
    """
    Function to get the coverpoint and cross dictionary for covergroup list
    Input Args: Beautiful Soup object of the covergroup html file
    Return val: dictionary summarizing the covergroup results
    """
    divs = c_grp.find_all('body')[0].find_all('div', recursive=False)
    cg_hier_path = divs[0].find('center', {'class': 'pagetitle'}).string
    cg_hier_path_list=cg_hier_path.split(':')
    cg_hier_path_list=[i.lstrip() for i in cg_hier_path_list if i]
    cg_hier_path=cg_hier_path_list[1]
    cg_name = cg_hier_path_list[2]

    ## now work on the cg summary table
    dict=extract_summary(divs[1])

    cp_info_list=[]
    cp_info_list_temp = extract_cp_details(divs[2])
    for cp in cp_info_list_temp:
        for summary in dict[2]:
            if summary !={}:
                if summary['VARIABLE'] == cp['name']:
                    cp['percent'] = summary['PERCENT']
                    cp['cross'] = 0
        for summary in dict[3]:
            if summary !={}:
                if summary['CROSS'] == cp['name']:
                    cp['percent'] = summary['PERCENT']
                    cp['cross'] = 1
        
        cp_info_list.append(cp)
    
    return {"name": str(cg_name), "coverpoints":cp_info_list}

def main():

    parser=argparse.ArgumentParser()
    parser.add_argument("-o", "--out_path", help="Path to Where the out folder")
    parser.add_argument("-yn", "--yaml_name", help="name of Yaml file")
    args = parser.parse_args()
    if os.path.isdir(args.out_path):
        _OUT_DIR= args.out_path
    else:
        print("Path  Doesn't exist!!")
        exit
    final_dict = {"covergroup":[]}
    for files in os.listdir(_OUT_DIR):
        if os.path.isfile(os.path.join(_OUT_DIR,files)) and re.match(r"grp\d+.html",files):
            with open(os.path.join(_OUT_DIR, "grp0.html"), 'r') as fp:
                c_grp=bs(fp, 'html.parser')
                final_dict['covergroup'].append(get_cg_dict_list(c_grp))

    with open(f"{args.yaml_name}", 'w') as dump:
        yaml.safe_dump(final_dict,dump)


if __name__=="__main__":
    main()