---
layout: post
title: WikiData and SPARQL queries
date: 2023-01-19 00:00:00
tags: sparql wikidata reference-post
categories: [blog]
comments: true
disqus_identifier: 20230119
preview_pic: /assets/images/2023-01-19-SPARQL_WikiData.png
---

A collection of several SPARQL queries to WikiData. Those are queries over different domains or topics that I've used for different goals. I've just decided to make them here public, so that I can quickly refer and reused them for new queries.

<br>

__Selects all the companies in the DAX TecDAX, MDAX and CDAX, gets their headquarters location's latitude and longitude and plots them in a map__ 

<script src="https://gist.github.com/davidsbatista/365d09fb6578c6b0a73dae0a0d2a3f81.js"></script>

[Try it](https://query.wikidata.org/#%23defaultView%3AMap%0ASELECT%20DISTINCT%20%3FcompanyLabel%20%3Fcompany%20%3Fheadquarters%20%3FheadquartersLabel%20%3Fcoordinates%20WHERE%20%7B%20%20%0A%20%20VALUES%20%3Fstock_markets%20%7Bwd%3AQ155718%20wd%3AQ378967%20wd%3AQ595622%20wd%3AQ874430%7D%20.%0A%20%20%3Fcompany%20wdt%3AP361%20%3Fstock_markets%20.%0A%20%20%3Fcompany%20rdfs%3Alabel%20%3FcompanyLabel%20.%20FILTER%28LANG%28%3FcompanyLabel%29%20%3D%20%22en%22%29%0A%20%20%3Fcompany%20wdt%3AP159%20%3Fheadquarters.%0A%20%20%3Fheadquarters%20rdfs%3Alabel%20%3FheadquartersLabel%20.%20FILTER%28LANG%28%3FheadquartersLabel%29%20%3D%20%22en%22%29%20%0A%20%20%3Fheadquarters%20wdt%3AP625%20%3Fcoordinates%0A%7D%0AORDER%20BY%20ASC%28%3FcompanyLabel%29
){:target="_blank"}

---

<br>

__Get countries and corresponding capital in English and German from Wikidata__

<script src="https://gist.github.com/davidsbatista/418b8dbe93d7f436f78d656e4e93541e.js"></script>

[Try it](https://query.wikidata.org/#SELECT%20%3Fcountry%20%3Fcountry_label%28lang%28%3Fcountry_label%29%20as%20%3Fcountry_label_lang%29%20%3Fcapital_label%28lang%28%3Fcapital_label%29%20as%20%3Fcapital_label_lang%29%20WHERE%20%7B%0A%20%20%3Fcountry%20wdt%3AP31%20wd%3AQ6256%3B%0A%20%20%20%20%20%20%20%20%20%20%20rdfs%3Alabel%20%3Fcountry_label%3B%0A%20%20%20%20%20%20%20%20%20%20%20wdt%3AP36%20%3Fcapital.%0A%20%20%3Fcapital%20rdfs%3Alabel%20%3Fcapital_label.%0A%20%20FILTER%28%20LANG%28%3Fcountry_label%29%20%3D%20%22de%22%20%7C%7C%20LANG%28%3Fcountry_label%29%20%3D%20%22en%22%29.%0A%20%20FILTER%28%20LANG%28%3Fcapital_label%29%20%3D%20%22de%22%20%7C%7C%20LANG%28%3Fcapital_label%29%20%3D%20%22en%22%29.%0A%7D%0AORDER%20BY%20ASC%28%3Fcountry_label%29){:target="_blank"}

---

<br>

__Get airports and cities served in English and German from Wikidata__

<script src="https://gist.github.com/davidsbatista/bf103dfcb0cbc64741bc821809f70525.js"></script>

[Try it](https://query.wikidata.org/#SELECT%20%3Fiata_code%20%3Fplaces_served_label%28LANG%28%3Fplaces_served_label%29%20AS%20%3Fplaces_served_label_lang%29%20%3Fcountry_code%0AWHERE%20%7B%0A%20%20%3Fitem%20wdt%3AP238%20%3Fiata_code.%0A%20%20%3Fitem%20wdt%3AP931%20%3Fplaces_served.%0A%20%20%3Fplaces_served%20rdfs%3Alabel%20%3Fplaces_served_label.%0A%20%20%3Fplaces_served%20wdt%3AP17%20%3Fcountry.%0A%20%20%3Fcountry%20wdt%3AP297%20%3Fcountry_code.%0A%20%20FILTER%28%20LANG%28%3Fplaces_served_label%29%20%3D%20%22de%22%20%7C%7C%20LANG%28%3Fplaces_served_label%29%20%3D%20%22en%22%29.%20%20%20%20%20%20%20%0A%7D%0ALIMIT%201000%0A){:target="_blank"}

---

<br>

__Capitals of counties in the USA, with state and state code, filtered by population__

<script src="https://gist.github.com/davidsbatista/3f9310a25274b3e2063bee3e1f5f877d.js"></script>

[Try it](https://query.wikidata.org/#SELECT%20DISTINCT%20%3Fcapital%20%3Fcapital_label%20%3Fpop%20%3Fcode%20WHERE%20%7B%0A%20%20%20%20%20%20%20%20%3Fcounty%20wdt%3AP31%2Fwdt%3AP279%2a%20wd%3AQ47168%20.%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%3Fcounty%20wdt%3AP36%20%3Fcapital%20.%0A%20%20%20%20%20%20%20%20%3Fcapital%20rdfs%3Alabel%20%3Fcapital_label%20.%0A%20%20%20%20%20%20%20%20%3Fcapital%20wdt%3AP1082%20%3Fpop%20.%0A%20%20%20%20%20%20%20%20%3Fcounty%20wdt%3AP131%20%3Fstate%20.%0A%20%20%20%20%20%20%20%20%3Fstate%20wdt%3AP31%20wd%3AQ35657%20.%20%0A%20%20%20%20%20%20%20%20%3Fstate%20wdt%3AP300%20%3Fcode%20.%0A%20%20%20%20%20%20FILTER%28LANG%28%3Fcapital_label%29%20%3D%20%22en%22%29%0A%20%20%20%20%20%20FILTER%28%3Fpop%20%3E%3D%2050000%29.%0A%0A%20%20%20%20%7D%20ORDER%20BY%20DESC%28%3Fpop%29){:target="_blank"}

---

<br>

__Number of connections served for a given airport__

<script src="https://gist.github.com/davidsbatista/ea5bf3a984cfa6e5ec0f27440a008f04.js"></script>

[Try it](https://query.wikidata.org/#SELECT%20%3Fiata_code%20%3Fairport_name%20%28COUNT%28%3Fconnectsairport%29%20AS%20%3Fnr_connections%29%20%0AWHERE%0A%7B%0A%20%20VALUES%20%3Fairport%20%7B%20wd%3AQ17480%20wd%3AQ9694%20wd%3AQ160556%20wd%3AQ403671%7D%0A%20%20%3Fairport%20wdt%3AP238%20%3Fiata_code.%0A%20%20%3Fairport%20rdfs%3Alabel%20%3Fairport_name.%0A%20%20OPTIONAL%20%7B%0A%20%20%20%20%20%20%3Fairport%20wdt%3AP81%20%3Fconnectsairport.%0A%20%20%7D%20%20%0A%20%20FILTER%28LANG%28%3Fairport_name%29%20%3D%20%22en%22%29%0A%7D%0AGROUP%20BY%20%3Fiata_code%20%3Fairport_name){:target="_blank"}

---

<br>


__Get all dams in Portugal with the latitude and longitude__
 
 
<script src="https://gist.github.com/davidsbatista/3eca48c03865413f724fb703dea49244.js"></script>
 
[Try it](https://query.wikidata.org/#%23defaultView%3AMap%0ASELECT%20DISTINCT%20%3Fdam%20%3Fcoords%20%3Flat%20%3Flong%20WHERE%20%7B%0A%20%20%20%20%20%20%20%20%3Fdam%20wdt%3AP31%2Fwdt%3AP279%2a%20wd%3AQ12323%20.%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%3Fdam%20wdt%3AP17%20wd%3AQ45%20.%0A%20%20%20%20%20%20%20%20%3Fdam%20p%3AP625%20%3Fcoordinataes%20.%0A%20%20%20%20%20%20%20%20%3Fcoordinataes%20ps%3AP625%20%3Fcoords%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20psv%3AP625%20%5B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20wikibase%3AgeoLatitude%20%3Flat%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20wikibase%3AgeoLongitude%20%3Flong%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5D%20.%0A%7D%20){:target="_blank"}

---

<br>

__Electric power dams in Portugal__

<script src="https://gist.github.com/davidsbatista/30fdead90869539114bb892c2f22ab6c.js"></script>
 
[Try it](https://query.wikidata.org/#%23defaultView%3AMap%0ASELECT%20DISTINCT%20%3Fdam%20%3Fname%20%3Felevation%20%3Fcoords%20WHERE%20%7B%0A%20%20%20%20%20%20%20%20%23%20%3Fdam%20wdt%3AP31%2Fwdt%3AP279%2a%20wd%3AQ12323%20.%0A%20%20%20%20%20%20%20%20%3Fdam%20wdt%3AP31%2Fwdt%3AP279%2a%20wd%3AQ15911738%20.%20%20%23%20only%20hidroelectric%20cpower%0A%20%20%20%20%20%20%20%20%3Fdam%20wdt%3AP17%20wd%3AQ45%20.%20%20%23%20only%20located%20in%20Portugal%0A%20%20%20%20%20%20%20%20%3Fdam%20rdfs%3Alabel%20%3Fname%20%20FILTER%28LANG%28%3Fname%29%20%3D%20%22pt%22%29%20.%0A%20%20%20%20%20%20%20%20%3Fdam%20p%3AP2044%20%3Felevation_sea_level%20.%0A%20%20%20%20%20%20%20%20%3Felevation_sea_level%20ps%3AP2044%20%3Felevation%20.%0A%20%20%20%20%20%20%20%20%3Fdam%20p%3AP625%20%3Fcoordinataes%20.%0A%20%20%20%20%20%20%20%20%3Fcoordinataes%20ps%3AP625%20%3Fcoords%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20psv%3AP625%20%5B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20wikibase%3AgeoLatitude%20%3Flat%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20wikibase%3AgeoLongitude%20%3Flong%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5D%20.%0A%7D%20){:target="_blank"}
