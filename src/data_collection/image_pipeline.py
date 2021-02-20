import hashlib

from scrapy.pipelines.images import ImagesPipeline
from scrapy.utils.python import to_bytes

class MultiDatasetPipeline(ImagesPipeline):
    def file_path(self, request, response=None, info=None, *, item=None):
        image_guid = hashlib.sha1(to_bytes(request.url)).hexdigest()
        return f"{info.spider.name}/{image_guid}.jpg"
