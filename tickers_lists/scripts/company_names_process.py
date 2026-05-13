import requests

from rapidfuzz import process, fuzz


def clean_company_name(company_name):
    """
    Simple cleaning of the end of the company (Co., Inc., etc)
    """

    suffixes = ['.', 'Inc', 'PLC', 'plc', 'Group', 'Corporation', 'Corp', 'Co', 'Company', 'S.A', 'L.P', 'Ltd', 'Limited', 'Group', ', ']
    company_name = company_name.strip()
    for suffix in suffixes:
        if company_name.endswith(suffix):
            company_name = company_name[:-len(suffix)]

    return company_name.strip()


def get_wikipedia_redirects(company_name):
    """
    Query wikipedia API to get redirects and urls (for DBPedia resource)
    """

    base_url = "https://en.wikipedia.org/w/api.php"

    company_names = [company_name, clean_company_name(company_name)]
        
    # The API can handle up to 50 titles per request
    params = {
        "action": "query",
        "titles": "|".join(company_names),
        "prop": "redirects|info", # Get pages that link to these
        "rdlimit": "max",
        "format": "json",
        "inprop": "url"
    }

    headers = {
    'User-Agent': 'CompanyDataBot/1.0 (emalherbe@example.com)'
    }

    urls = []
    response = requests.get(base_url, params=params, headers=headers)
    if response.ok:
        response = response.json()
        print(response)

        pages = response.get("query", {}).get("pages", {})
        
        results = {}
        for page_id, content in pages.items():
            urls.append(content.get("canonicalurl"))
            title = content.get("title")
            # List of titles that redirect to this canonical page
            redirects = [r["title"] for r in content.get("redirects", [])
                         if not ('Wikipedia talk' in r["title"]
                                 or '(' in r['title'] # Typically avoid (disambiguation) pages
                                 or r['title'].startswith('User:')
                                 or r['title'].startswith('Draft:'))
                        ]
            company_names.extend(redirects)
            print('redirects', redirects)
    else:
        print('error', response)

    return list(set(company_names)), urls


def get_company_dbpedia_names(resource_name):
    """
    Query DBPedia API to get redirects, label, name.
    
    resource_name: last part of wikipedia url resource
    """

    # Don't take dbo:wikiPageRedirects from DBpedia: 
    # https://dbpedia.org/page/Allison_Transmission has shit redirects...
    url = "https://dbpedia.org/sparql"

    # Possibility to filter on entity type with: ?company a ?type . (on dbo:company)
    # But not so efficient in my quick tests. To dig further if too much noisy names
    query = f"""
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dbo: <http://dbpedia.org/ontology/>
    
    SELECT ?label ?name (GROUP_CONCAT(DISTINCT ?altName; separator="|") AS ?redirects) WHERE {{
      BIND(<http://dbpedia.org/resource/{resource_name}> AS ?company)
      
      ?company rdfs:label ?label .
      
      OPTIONAL {{ ?company foaf:name ?name }} .
      
      # Find resources that redirect TO this company
      OPTIONAL {{ 
        ?altName dbo:wikiPageRedirects ?company .
      }}
      
      FILTER (lang(?label) = "en")
    }} 
    GROUP BY ?label ?name
    """

    params = {
        "query": query,
        "format": "application/sparql-results+json"
    }
    
    response = requests.get(url, params=params)
    
    # Handle potential 403 or 500 errors
    if response.status_code != 200:
        return f"Error: {response.status_code}"
        
    data = response.json()
    print(data)
    results = data.get("results", {}).get("bindings", [])

    names = []
    for results in results:
        for key in ['name', 'label', 'redirects']:
            name = results.get(key, {}).get('value', '')
            if name:
                if key == 'redirects':
                    new_names = name.replace('http://dbpedia.org/resource/', '').replace('_', ' ').split('|')
                    names.extend([name for name in new_names if not '(' in name])
                else:
                    names.append(name)

    return names


def find_overlaps(base_name, alt_names, threshold=50):
    """
    Using base name, filter the alternative names to remove noisy string that are too different.
    """

    # process.extract returns (match, score, index)
    results = process.extract(
        base_name, 
        alt_names, 
        scorer=fuzz.WRatio, # WRatio handles case and partial strings well
        score_cutoff=threshold,
        limit=None
    )
    
    return [name for name, _, _ in results]
