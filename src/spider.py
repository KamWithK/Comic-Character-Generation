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
