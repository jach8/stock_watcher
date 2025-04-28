#made using https://github.com/goldsmith/Wikipedia

import wikipedia
import urllib
import json
import threading

def main():
    open("logos/notfound.txt", 'w').close()
    open("logos/found.txt", 'w').close()
    brandNames = [line.strip() for line in open('brands.txt')]
    for brand in brandNames:
        thr = threading.Thread(target=getCompanyLogo, args = [brand])
        thr.start()

def getCompanyLogo(companyName):
    try:
        logoURL = None
        # if logoURL == None:
        #     #using google as a fallback
        #     # print(companyName+ " logo")
        #     # logoURL = google_image(companyName+" logo")
        #     print(f'{companyName} Check 1')
        if logoURL == None:
            logoURL = getWikiLogoURL(companyName)
            add_to_found(companyName, logoURL)
        if logoURL == None:
            addToNotFoundList(companyName)
        return
        # urllib.urlretrieve(logoURL, "logos/"+companyName+getExtension(logoURL))
    except:
        addToNotFoundList(companyName)
        return

#to do ensure its in the class logo
def getWikiLogoURL(pageName):
	try:
		myPage = wikipedia.page(pageName)
		imageUrls = myPage.images
		logoLinks = []
		for url in imageUrls:
			lowerCaseUrl = url.lower()
			if "commons-logo" not in lowerCaseUrl and "-logo" not in lowerCaseUrl and "_logo" in lowerCaseUrl and ".svg" in lowerCaseUrl:
				logoLinks.append(url)
		print(logoLinks)
		return logoLinks[-1]
	except:
		return None

    
def getExtension(url):
	for i in range(len(url)-1, -1, -1):
		if url[i] == '.':
			return url[i:]

def addToNotFoundList(companyName):
	f = open("logos/notfound.txt", 'a')
	f.write(companyName+"\n")
	f.close()

def add_to_found(companyName, logoURL):
    f = open("logos/found.txt", 'a')
    f.write(companyName+","+logoURL+"\n")
    f.close()


main()