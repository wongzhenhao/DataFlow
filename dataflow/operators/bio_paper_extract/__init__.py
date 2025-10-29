from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # filter

    # eval

    # generate
    from .generate.paper_downloader_generator import PaperDownloaderGenerator
    from .generate.paper_parsing_generator import PaperParsingGenerator
    from .generate.paper_info_extract_generator import PaperInfoExtractGenerator

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/bio_paper_extract/"

    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/bio_paper_extract/", _import_structure)
