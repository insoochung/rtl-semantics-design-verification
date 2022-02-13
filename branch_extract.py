#!/usr/bin/python3
import os
import sys
import argparse
from bs4 import BeautifulSoup
import re

parser=argparse.ArgumentParser()
parser.add_argument("-cp", "--covPath", help="Coverage HTML path", required=False)
parser.add_argument("-cd", "--covDir", help="Path to the urgReport which contains all the coverage HTML", required=False)

#Get the argument list
args = parser.parse_args()
#path to where all the coverage html files are
'''TODO Change the path to read from a commandline'''
if args.covPath is not None:
    file_path = args.covPath #"/home/abuturab/ML-In-DV/alu_cov_report/urgReport/mod37.html"
else:
    file_path=False
if args.covDir is not None:
    dir_path=args.covDir
else:
    dir_path=False

def coverage_all_files():
    if os.path.isdir(dir_path) and not os.path.isdir(os.path.join(os.getcwd(),"CP_dump")):
        os.mkdir(os.path.join(os.getcwd(),"CP_dump"))

    for file in os.listdir(dir_path):
        if re.search(r".*.html$", file) and os.path.isfile(os.path.join(dir_path,file)):
            srch_temp=re.search(r"(.*).html", file)
            cov_name=srch_temp.group(1)
            file_path=os.path.join(dir_path,file)
            # print("file path:", file_path)
            # print("file name: ", "."+cov_name+".yaml")
            if write_branch_to_temp(os.path.join(dir_path,file))==1:
                print("Branch found in file:", file_path)
                branch_dict=set_branch_dict()
                write_yaml_file(branch_dict, cov_name)
                cleanup_temp_files()
            else:
                cleanup_temp_files()         

def create_branch_temp():
    try:
        file_temp=open(os.path.join(os.getcwd(),".cov_temp.html"),"w+")
    except:
        print("ERROR OPENING TEMP FILE !!")
        exit
    return file_temp

def create_branch_table_temp():
    try:
        file_temp=open(os.path.join(os.getcwd(),".cov_table_temp.html"),"w+")
    except:
        print("ERROR OPENING TEMP FILE !!")
        exit
    return file_temp

def write_branch_to_temp(html_file):
    _BRANCH_START_FLAG=0
    _BRANCH_FOUND=0
    #file_path = args.covPath
    write_file_pointer=create_branch_temp()
    write_table_file_pointer=create_branch_table_temp()

    print("Working on File:", html_file)
    if os.path.isfile(html_file):
        with open(html_file,'r') as file:
            for line in file:
                if re.search(r"Branch Coverage for",line):
                    _BRANCH_START_FLAG=1
                    _BRANCH_FOUND=1
                if re.search(r"pre class", line) and _BRANCH_START_FLAG==1:
                    _BRANCH_START_FLAG=2
                if re.search(r"Assert Coverage for",line) or re.search(r"Line Coverage for",line):
                    _BRANCH_START_FLAG=0
                    #print(line)
                if _BRANCH_START_FLAG==1:
                    write_table_file_pointer.write(line)
                if _BRANCH_START_FLAG==2:
                    write_file_pointer.write(line)
    write_file_pointer.close()
    write_table_file_pointer.close()
    return _BRANCH_FOUND

def set_branch_dict():

    branch_dict={'TYPE':[], 'LINE':[], 'SCORE':[]}

    #Open the temporary file which has the coverage table
    with open(os.path.join(os.getcwd(),".cov_table_temp.html"),"r") as file:
        cov_read=BeautifulSoup(file.read(), 'html.parser')

    tables = cov_read.find_all("table")
    #head_row=tables[0].find("tr").find_all("th")
    #print(tables[0])
    #print("Branches", end=",")
    # for entry in head_row:
    #     print(entry.string, end=",")
    # print()

    for table in tables:
        row = table.find_all("tr")
        for irow in row:
            entries=irow.find_all("td")
            if len(entries) > 0:
                branch_dict['TYPE'].append(entries[0].string)
                branch_dict['LINE'].append(entries[1].string)
                branch_dict['SCORE'].append(entries[4].string)
    return branch_dict

def write_yaml_file(branch_dict, cov_name="default_cov"):
    with open(os.path.join(os.getcwd(),".cov_temp.html"),"r") as file:
        cov_read=BeautifulSoup(file.read(), 'html.parser')
    
    write_dir=os.path.join(os.getcwd(),"CP_dump")
    if not os.path.isdir(write_dir):
        os.mkdir(write_dir)
    
    fp=open(os.path.join(write_dir,"."+cov_name+".yaml"),"w") #as file:
    print(os.path.join(write_dir,"."+cov_name+".yaml"))
    code = cov_read.find_all("pre")
    tables = cov_read.find_all("table")
    count=0
    print(branch_dict)
    for table in tables:
        if count<len(code):
            line_no=re.search(r"([\d]+)", str(code[count]))
            # print("Branch: ", count)
            # print("\tLine: ",line_no.group(1))
            # print("\tPath:")
            fp.write("Branch: "+str(count)+"\n")
            fp.write("\tLine: "+str(line_no.group(1))+"\n")
            print(str(line_no.group(1)))
            print(branch_dict['LINE'].index(str(line_no.group(1))))
            fp.write("\tType: "+branch_dict['TYPE'][branch_dict['LINE'].index(str(line_no.group(1)))]+"\n")
            fp.write("\tPaths:")       
            count+=1
        rows = table.find_all("tr")
        for irow in rows:
            entries=irow.find_all("td")
            if len(entries)>0:
                #print("\t\t",end="- ")
                fp.write("\t\t- Trace: ")
            for entry in entries:
                entry_str=str(entry.string)
                #print(entry_str,end=":")
                #print(entry_str=='Covered')
                if entry_str != 'Covered' and entry_str !='Not Covered' and entry_str != '-':
                    # print(entry_str, end=",")
                    # print(entry_str=='Covered')
                    fp.write(re.sub(r"(?:CASEITEM.*:\ |\ )", ":",entry_str.strip())+",")
                elif entry_str=='-':
                    fp.write("X,")
                elif entry_str == 'Covered':
                    #fp.write("_1_")
                    fp.write("\n\t\t  Status: Covered")
                elif entry_str == 'Not Covered':
                    #fp.write("_0_")
                    fp.write("\n\t\t  Status: Not_Covered")
            fp.write("\n")
        fp.write("\n") 
    fp.close()

def cleanup_temp_files():
    if os.path.exists(".cov_temp.html"):
        os.remove(".cov_temp.html")
    if os.path.exists(".cov_table_temp.html"):
        os.remove(".cov_table_temp.html")

if dir_path != False:
    coverage_all_files()
elif file_path !=False:
    write_branch_to_temp(file_path)
    branch_dict=set_branch_dict()
    write_yaml_file(branch_dict)
    cleanup_temp_files()
