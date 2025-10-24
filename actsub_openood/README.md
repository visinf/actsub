## Evaluation and Tuning

To evaluate ActSub on the OpenOOD benchmark, run the following:

```bash
python evaluate_actsub.py --config actsub_common_config.yml
```

Modify the parameters in **`actsub_common_config.yml`** to select different backbones and experimental variants. The same script can also tune the parameters and apply them before evaluation if tuning is enabled in the configuration. For method specific configurations, refer to:

- **`configs/postprocessors/actsub.yml`** The decisive component uses either [SCALE](https://github.com/kai422/SCALE) or [ASH-S](https://github.com/andrijazz/ash).
- **`configs/postprocessors/actsub_gen.yml`** The decisive component uses [GEN](https://github.com/XixiLiu95/GEN).

The **`is_gen field`** in **`actsub_common_config.yml`** determines which of the two configs file will be used for the evaluation. For the detailed configuration setting refer to **`configs/postprocessors/actsub.yml`** and **`configs/postprocessors/actsub_gen.yml`**.
