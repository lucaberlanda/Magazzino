import xml.etree.ElementTree as ET

# https://stackoverflow.com/questions/20999239/elementtree-find-findall-cant-find-tag-with-namespace


def print_ptf_holdings(full_holdings, isin_only=False):
    cmm_str = '{http://schemas.thomsonreuters.com/2012/06/30/df5v1.0}'

    if isin_only:
        holding_str = ''
        for holding in full_holdings:
            for i in holding.iter():
                if i.tag == cmm_str + 'CrossReferenceCodes':
                    holding_codes = i.findall('.//' + cmm_str + 'Code')
                    for holding_code in holding_codes:
                        if holding_code.attrib['Type'] == 'ISIN':
                            holding_str += holding_code.text + ', '

        print(holding_str)

    else:
        for holding in full_holdings:
            holding_str = ''
            for i in holding.iter():

                if i.tag == cmm_str + 'CrossReferenceCodes':
                    holding_codes = i.findall('.//' + cmm_str + 'Code')
                    for holding_code in holding_codes:
                        if holding_code.attrib['Type'] == 'ISIN':
                            holding_str += ' ISIN: ' + holding_code.text + ';'

                if i.tag == cmm_str + 'Name':
                    holding_str += 'Name: ' + i.text + ';'

                if i.tag == cmm_str + 'Weight':
                    holding_str += ' Weight: ' + i.text + ';'

                if i.tag == cmm_str + 'MarketValueCurrency':
                    holding_str += ' FX: ' + i.text + ';'

                if i.tag == cmm_str + 'MaturityDate':
                    holding_str += ' Maturity: ' + i.text + ';'

                if i.tag == cmm_str + 'CouponRate':
                    holding_str += ' Coupon: ' + i.text + ';'

            print(holding_str)


def print_generic_info_and_get_full_ptf(root):
    # full_file = ET.tostring(root, encoding='utf8').decode('utf8')  # full file txt
    cmm_str = '{http://schemas.thomsonreuters.com/2012/06/30/df5v1.0}'
    codes = root.findall('.//' + cmm_str + 'CrossReferenceCodes')[0]
    for child in codes:
        if child.attrib['Type'] == 'ISIN Code':
            print(child.text)

    last_ptf_element = root.findall('.//' + cmm_str + 'PortfolioHistory')[0].findall('.//' + cmm_str + 'Portfolio')[-1]
    full_ptf = last_ptf_element.findall('.//' + cmm_str + 'Holdings')[0].findall('.//' + cmm_str + 'Holding')

    print('Last ptf date: ' + last_ptf_element.attrib['Date'])
    print('Number of securities: ' + str(len(full_ptf)))

    ratings = root. \
        findall('.//' + cmm_str + 'ShareClasses')[0]. \
        findall('.//' + cmm_str + 'ShareClass')[0]. \
        findall('.//' + cmm_str + 'Ratings')[0]  # what is this?

    return full_ptf


tree = ET.parse("35042108.xml")
rt = tree.getroot()
ptf = print_generic_info_and_get_full_ptf(root=rt)  # generic info
print_ptf_holdings(ptf, isin_only=False)  # print full holdings
