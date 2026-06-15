"""Quick and dirty script to use company_name_process routines and generate a json of company names.

Input  (`companies.md`):  one company per line, formatted `- `KEY` Company Name`
                          where KEY is any stable string used as the dict key
                          in the output JSON (e.g. `TICKER` or `TICKER|EXCHANGE`
                          to disambiguate cross-exchange ticker collisions).
                          The Company Name part is what gets fed to Wikipedia /
                          DBPedia for alias lookup.
Output (`company_alt_names.json`):  {KEY: [list of alt names]}
"""

import json

from company_names_process import clean_company_name, get_wikipedia_redirects, get_company_dbpedia_names, find_overlaps


with open('companies.md') as f:
    companies = f.read().splitlines()

companies_by_key = {
    company.split("`", 2)[1]: company.split("`", 2)[2].strip()
    for company in companies if company.startswith("- `")
}

names_by_key = dict()
for key, company in companies_by_key.items():
    names, urls = get_wikipedia_redirects(company)
    print('URL', urls)
    for url in urls:
        dbpedia_resource = url.split('/')[-1]
        dbpedia_names = get_company_dbpedia_names(dbpedia_resource)
        names.extend(dbpedia_names)
    print(names)
    names = find_overlaps(company, names, threshold=60)
    names_by_key[key] = list(set(names))

with open('company_alt_names.json', 'w') as f:
    json.dump(names_by_key, f)
