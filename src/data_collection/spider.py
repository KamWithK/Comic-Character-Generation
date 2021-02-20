import scrapy

from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

class SuperheroSpider(CrawlSpider):
    name = "superhero"
    start_urls = ["https://superhero-database.weebly.com/a-z-database.html"]
    rules = [Rule(LinkExtractor(allow="https://superhero-database\.weebly\.com/[a-z]\.html"), callback="parse")]

    def parse(self, response):
        image_urls = [response.urljoin(image_link) for image_link in response.css(".wsite-image::attr(src)").getall()]
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
