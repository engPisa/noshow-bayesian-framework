import pandas as pd
import pytest

from noshow.data import load_raw


@pytest.mark.parametrize("target", [None, "talvez"])
def test_alvo_invalido_nao_vira_comparecimento(tmp_path, target):
    path = tmp_path / "raw.csv"
    pd.DataFrame({"no_show": [target], "appointment_date": ["01/01/2020"],
                  "date_of_birth": ["01/01/2000"], "entry_service_date": ["01/01/2019"],
                  "appointment_month": ["jan"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no_show ausente ou inválido"):
        load_raw(path)
