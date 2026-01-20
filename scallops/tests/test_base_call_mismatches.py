import pandas as pd
import pytest

from scallops.reads import summarize_base_call_mismatches

barcodes_df = pd.DataFrame(data=dict(barcode=["AAAAA", "CCCCC", "GGGGG"]))


@pytest.mark.basecalls
def test_summarize_base_call_mismatches_no_matches():
    reads_df = pd.DataFrame(data=dict(barcode=["AAAAA", "AAAAA", "GGGGG"]))
    reads_df["label"] = 1
    assert len(summarize_base_call_mismatches(reads_df, barcodes_df)) == 0


@pytest.mark.basecalls
def test_summarize_base_call_mismatches():
    reads_df = pd.DataFrame(data=dict(barcode=["AAAAG", "GGCGG", "AAAAG", "AAGAG"]))
    reads_df["label"] = 1
    result_df = summarize_base_call_mismatches(reads_df, barcodes_df)
    assert len(result_df) == 2
    assert (
        len(
            result_df.query(
                'called_base=="C" & whitelist_base=="G" & read_position==2 & count==1'
            )
        )
        == 1
    )
    assert (
        len(
            result_df.query(
                'called_base=="G" & whitelist_base=="A" & read_position==4 & count==2'
            )
        )
        == 1
    )
