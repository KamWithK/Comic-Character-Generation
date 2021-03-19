from scrapy.crawler import CrawlerProcess
from spider import SuperheroSpider, DCSpider, PDSHSpider, MarvelSpider

process = CrawlerProcess(settings={
    "COOKIES_ENABLED": "False",
    "LOG_LEVEL": "WARNING",
    "ITEM_PIPELINES": {"image_pipeline.MultiDatasetPipeline": 1},
    "IMAGES_STORE": "data"
})

process.crawl(SuperheroSpider)
process.crawl(DCSpider)
process.crawl(MarvelSpider)
process.crawl(PDSHSpider)
process.start()
