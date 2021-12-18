import random

def random_network_creator(n):
    ''' Creates a random network for a given number of variables '''

    # create a new network file 
    file_name = "random_networks/demofile2.BIFXML" # can change output filename here
    f = open(file_name, "a")

    # write stock variables into file
    f.writelines([
        '<?xml version="1.0" encoding="US-ASCII"?>' '\n',
        '<!DOCTYPE BIF [''\n', 
        '	      <!ATTLIST BIF VERSION CDATA #REQUIRED>''\n', 
        '	<!ELEMENT NETWORK ( NAME, ( PROPERTY | VARIABLE | DEFINITION )* )>''\n',
        '	<!ELEMENT NAME (#PCDATA)>''\n',
        '	<!ELEMENT VARIABLE ( NAME, ( OUTCOME |  PROPERTY )* ) >''\n',
        '	      <!ATTLIST VARIABLE TYPE (nature|decision|utility) "nature">''\n',
        '	<!ELEMENT OUTCOME (#PCDATA)>''\n',
        '	<!ELEMENT DEFINITION ( FOR | GIVEN | TABLE | PROPERTY )* >''\n',
        '	<!ELEMENT FOR (#PCDATA)>''\n',
        '	<!ELEMENT GIVEN (#PCDATA)>''\n',
        '	<!ELEMENT TABLE (#PCDATA)>''\n',
        '	<!ELEMENT PROPERTY (#PCDATA)>''\n',
        ']>''\n',
        '\n',
        '\n',
        '<BIF VERSION="0.3">''\n',
        '<NETWORK>''\n',
        '<NAME>random-network</NAME>''\n',
        '\n',
        '<!-- Variables -->',
        '\n'])
    # write n variables into the file        
    i = 0
    while i < n:
        f.writelines([
            '<VARIABLE TYPE="nature">' '\n',
            '	<NAME>NODE_'+str(i)+'</NAME>''\n', 
            '	<OUTCOME>true</OUTCOME>''\n', 
            '	<OUTCOME>false</OUTCOME>''\n',
            '	<PROPERTY>position = ('+str(random.randint(-500, 500))+', '+str(random.randint(-500, 500))+')</PROPERTY>''\n', #dont think this is used atm
            '</VARIABLE>''\n',
            '\n'])
        i += 1
    f.writelines(['<!-- Probability distributions -->',
    '\n'])
    # NEXT: write n probability distributions for all the variables
    i = 0
    lst = list(range(0, n))
    random.shuffle(lst)
    while i < n:
        var_1 = lst.pop() # guarantees all the nodes in the network are present at least once (never any leaf nodes like this)
        var_2 = random.randint(0, n)
        var_3 = random.randint(0, n)
        var_4 = random.randint(0, n)
        if var_1 == var_2:
            var_2 = random.randint(0, n)
        if var_1 == var_3:
            var_3 = random.randint(0, n)
        if var_1 == var_4:
            var_4 = random.randint(0, n)

        # could make random cpt values, but i don't currently see the use for the purpose of random networks
        f.writelines([
            '<DEFINITION>' '\n',
            '	<FOR>NODE_'+ str(var_1) +'</FOR>' '\n',
            #'	<FOR>NODE_'+ str(var_3) +'</FOR>' '\n', # the second <FOR> connection can be removed along with the CPT values, because less edges will be drawn
            '	<GIVEN>NODE_'+ str(var_2) +'</GIVEN>' '\n'
            #'	<GIVEN>NODE_'+ str(var_4) +'</GIVEN>' '\n' # this can also be removed if there is less edges required
            '	<TABLE>0.6 0.4 0.05 0.95 </TABLE>' '\n',
            '</DEFINITION>' '\n',
            '\n'])
        i += 1

    # closing off the document properly
    f.writelines([
        '</NETWORK>' '\n',
        '</BIF>'])

    f.close()

random_network_creator(15) # change size of the network to be created here
