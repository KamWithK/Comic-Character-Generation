import scrapy
import re

from scrapy import Spider
from scrapy.spiders import CrawlSpider, XMLFeedSpider, Rule
from scrapy.linkextractors import LinkExtractor

class SuperheroSpider(CrawlSpider):
    name = "superhero"
    start_urls = ["https://superhero-database.weebly.com/a-z-database.html"]
    rules = [Rule(LinkExtractor(allow="https://superhero-database\.weebly\.com/[a-z]\.html"), callback="parse")]

    def parse(self, response):
        image_urls = [response.urljoin(image_link) for image_link in response.css(".wsite-image::attr(src)").getall()]
        yield {"image_urls": image_urls}

class DCSpider(Spider):
    name = "dccomics"
    start_urls = ["https://www.dccomics.com/proxy/search?type=generic_character"]

    def parse(self, response):
        superheros = response.json()["results"]
        image_urls = [response.urljoin(superhero["fields"]["field_profile_picture:file:url"][0]) for superhero in superheros.values()]
        yield {"image_urls": image_urls}

class PDSHSpider(CrawlSpider):
    name = "pdsh"
    start_urls = ["https://pdsh.fandom.com/wiki/Category:Debut_Year"]
    rules = [
        Rule(LinkExtractor(restrict_css=".category-page__pagination-next")),
        Rule(LinkExtractor(restrict_css=".category-page__member-link"), callback="parse")
    ]

    def parse(self, response):
        image_urls = [url.split("/revision")[0] for url in response.css(".category-page__member-thumbnail::attr(src)").getall()]
        yield {"image_urls": image_urls}

class MarvelSpider(XMLFeedSpider):
    name = "marvel"
    start_urls = [
        "https://www.marvel.com/sitemap-8.xml",
        "https://www.marvel.com/sitemap-9.xml",
        "https://www.marvel.com/sitemap-10.xml",
        "https://www.marvel.com/sitemap-11.xml"
    ]
    itertag = 'url'

    def parse_node(self, response, node):
        character_url_search = re.search("https://www.marvel.com/characters/", node.extract())
        image_url_search = re.search("https:\/\/terrigen-cdn-dev\.marvel\.com\/content\/prod\/2x\/.*?\.(?:jpg|png)", node.extract())
        if character_url_search and image_url_search:
            yield {"image_urls": [image_url_search.group(0)]}