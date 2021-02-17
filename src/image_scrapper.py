from scrapy.crawler import CrawlerProcess
from spider import SuperheroSpider

process = CrawlerProcess(settings={
    "COOKIES_ENABLED": "False",
    "LOG_LEVEL": "WARNING",
    "ITEM_PIPELINES": {"scrapy.pipelines.images.ImagesPipeline": 1},
    "IMAGES_STORE": "data"
})

process.crawl(SuperheroSpider)
process.start()
