import networkx as nx
from typing import Union
from xml.etree.ElementTree import TreeBuilder
from numpy import multiply
from BayesNet import BayesNet
import copy
import pandas as pd
import random
from BNReasoner import BNReasoner

def random_network_creator(n):
    ''' Creates a random network for a given number of variables '''

    # create a new network file 
    file_name = "random_networks/demofile2.BIFXML"
    f = open(file_name, "a")

    # copy basic lines into file
    with open("random_networks/basic.BIFXML", "r") as scan:
        f.write(scan.read())
    

    # write variables into file
    f.writelines([
        '<VARIABLE TYPE="nature">',
            '<NAME>light-on</NAME>', 
            '<OUTCOME>true</OUTCOME>', 
            '<OUTCOME>false</OUTCOME>',
            '<PROPERTY>position = (73, 165)</PROPERTY>',
        '</VARIABLE>'])
    # NEXT: find out why this does not work
    
    f.close()
    
    #example_variables = ['light-on', 'bowel-problem','dog-out','hear-bark', 'family-out']
    #example_positions = [(73, 165), (190, 69), (155, 165), (154, 241), (112, 69)]
    #for variable in example_variables:
    #    print(variable)

    # open and read the file after the appending
    #f = open("random_networks/demofile2.BIFXML", "r")
    #print(f.read())

random_network_creator(1)