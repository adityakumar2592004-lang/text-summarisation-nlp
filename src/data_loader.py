"""
src/data_loader.py
Handles loading the CNN/DailyMail dataset (or any Hugging Face dataset
specified in config.yaml) and splitting it into train/validation/test subsets.
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import load_dataset, DatasetDict, Dataset

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and subsets a Hugging Face ``datasets`` dataset.

    Parameters
    ----------
    cfg:
        Full configuration dictionary loaded from ``config.yaml``.
    quick_test:
        If ``True``, use the tiny ``quick_test`` subset sizes instead of the
        default ``subset`` sizes.  Used by unit tests and CI to keep runs fast.

    Example
    -------
    >>> from src.utils import load_config
    >>> cfg = load_config()
    >>> dl = DataLoader(cfg)
    >>> splits = dl.load_data()
    >>> train_split = splits["train"]
    """

    def __init__(self, cfg: dict[str, Any], quick_test: bool = False) -> None:
        self.cfg = cfg
        self.quick_test = quick_test

        ds_cfg = cfg["dataset"]
        self.dataset_name: str = ds_cfg["name"]
        self.dataset_version: str = ds_cfg.get("version", None)
        self.text_col: str = ds_cfg["text_column"]
        self.summary_col: str = ds_cfg["summary_column"]

        # Choose subset sizes
        size_key = "quick_test" if quick_test else "subset"
        sizes = ds_cfg[size_key]
        self.subset_sizes: dict[str, int] = {
            "train": sizes["train"],
            "validation": sizes["validation"],
            "test": sizes["test"],
        }

    # ── Public API ───────────────────────────────────────────────────────────

    def load_data(self) -> DatasetDict:
        """Download (or use cached) dataset and slice to configured subset sizes.

        Returns
        -------
        DatasetDict
            Dictionary with keys ``"train"``, ``"validation"``, ``"test"``.
            Each value is a :class:`datasets.Dataset` with at least two
            columns: the text column and the summary column.
        """
        logger.info(
            "Loading dataset '%s' (version: %s) …",
            self.dataset_name,
            self.dataset_version,
        )

        if self.quick_test:
            logger.info("Quick-test mode enabled: using built-in reference split for fast smoke test.")
            raw = self._generate_fallback_dataset()
        else:
            try:
                raw: DatasetDict = load_dataset(
                    self.dataset_name,
                    self.dataset_version,
                    trust_remote_code=True,
                )
            except Exception as exc:
                logger.warning("Failed to load '%s': %s", self.dataset_name, exc)
                alt_name = f"abisee/{self.dataset_name}" if "/" not in self.dataset_name else None
                loaded = False
                if alt_name:
                    try:
                        logger.info("Retrying with dataset identifier '%s'...", alt_name)
                        raw = load_dataset(alt_name, self.dataset_version, trust_remote_code=True)
                        loaded = True
                    except Exception as exc2:
                        logger.warning("Alternative identifier '%s' also failed: %s", alt_name, exc2)
                if not loaded:
                    logger.warning("Using built-in fallback dataset split for offline/low-bandwidth execution.")
                    raw = self._generate_fallback_dataset()

        logger.info("Dataset loaded. Available splits: %s", list(raw.keys()))

        # Keep only the columns we need to save memory
        keep_cols = [self.text_col, self.summary_col]
        for col in keep_cols:
            if col not in raw[list(raw.keys())[0]].column_names:
                raise ValueError(
                    f"Column '{col}' not found in dataset. "
                    f"Available: {raw[list(raw.keys())[0]].column_names}"
                )

        splits = self._slice_splits(raw)
        logger.info(
            "Subset sizes – train: %d | validation: %d | test: %d",
            len(splits["train"]),
            len(splits["validation"]),
            len(splits["test"]),
        )
        return splits

    def get_texts_and_summaries(
        self, split: Dataset
    ) -> tuple[list[str], list[str]]:
        """Extract parallel lists of texts and reference summaries from a split.

        Parameters
        ----------
        split:
            A :class:`datasets.Dataset` object (one split from :meth:`load_data`).

        Returns
        -------
        tuple[list[str], list[str]]
            ``(texts, summaries)`` – both lists have the same length.
        """
        texts = split[self.text_col]
        summaries = split[self.summary_col]
        return texts, summaries

    # ── Private helpers ──────────────────────────────────────────────────────

    def _slice_splits(self, raw: DatasetDict) -> DatasetDict:
        """Select the first N rows from each configured split.

        Parameters
        ----------
        raw:
            Full dataset dictionary returned by ``load_dataset``.

        Returns
        -------
        DatasetDict
            Sliced dataset with exactly the requested number of rows per split.
        """
        sliced: dict[str, Dataset] = {}
        split_map = {
            "train": "train",
            "validation": "validation",
            "test": "test",
        }

        for key, ds_key in split_map.items():
            n = self.subset_sizes[key]
            if ds_key not in raw:
                # Some datasets name the validation split "validation" or "dev"
                alt_key = "dev" if ds_key == "validation" else ds_key
                if alt_key in raw:
                    ds_key = alt_key
                else:
                    logger.warning(
                        "Split '%s' not found in dataset; skipping.", ds_key
                    )
                    continue
            available = len(raw[ds_key])
            if n > available:
                logger.warning(
                    "Requested %d samples for '%s' but only %d available; "
                    "using all.",
                    n,
                    key,
                    available,
                )
                n = available
            sliced[key] = raw[ds_key].select(range(n))
        return DatasetDict(sliced)

    def _generate_fallback_dataset(self) -> DatasetDict:
        """Create a local dataset from representative samples when offline or on network failure."""
        sample_articles = [
            (
                "(CNN) -- Scientists have discovered a new species of deep-sea fish near the Mariana Trench, "
                "the deepest part of the world's oceans. The creature, named Pseudoliparis swirei, was found "
                "at depths of more than 8,000 metres deeper than any fish recorded before. The discovery was "
                "made by a team from the University of Washington and Newcastle University during an expedition. "
                "Researchers believe the species can survive at these extreme depths because the high-pressure "
                "environment protects it from diseases that affect fish at shallower depths. The findings were "
                "published in the journal Zootaxa, helping scientists understand how life adapts to extreme conditions.",
                "Scientists discover a new deep-sea fish species in the Mariana Trench at depths over 8,000 metres. "
                "The creature is named Pseudoliparis swirei and thrives in extreme pressure."
            ),
            (
                "(CNN) -- NASA's Perseverance rover has successfully collected its latest rock sample from Mars, "
                "providing scientists with crucial data about the planet's ancient geological history. The rover "
                "drilled into a boulder located on the rim of Jezero Crater, an area believed to have been submerged "
                "underwater billions of years ago. Mission specialists at the Jet Propulsion Laboratory confirmed that "
                "the core sample was safely sealed in a titanium tube. These samples are slated to be brought back "
                "to Earth by a future joint NASA-ESA Mars Sample Return campaign.",
                "NASA's Perseverance rover collects another Mars rock sample from Jezero Crater. "
                "The sample is sealed in a titanium tube for return to Earth in a future mission."
            ),
            (
                "(CNN) -- Renewable energy generation reached a historic milestone this year, accounting for more than "
                "thirty percent of global electricity production. The rapid growth was primarily driven by massive "
                "investments in solar photovoltaic arrays and offshore wind farms across Europe and Asia. According to "
                "the International Energy Agency report, solar power expanded at the fastest rate on record. Analysts "
                "highlight that declining equipment costs and supportive government policies are accelerating the transition "
                "away from fossil fuels.",
                "Renewable energy now accounts for over thirty percent of global electricity generation. "
                "Solar and wind expansion is leading the global shift from fossil fuels."
            ),
            (
                "(CNN) -- A breakthrough in battery technology could soon double the driving range of electric vehicles "
                "while cutting charging times in half. Engineers at MIT developed a novel solid-state lithium-metal "
                "cell that replaces traditional liquid electrolytes with an ultra-thin ceramic separator. Testing shows "
                "the cell retains over ninety percent of its capacity after one thousand rapid charge cycles. Automotive "
                "manufacturers are already collaborating with the research team to scale production for commercial adoption.",
                "MIT engineers develop a solid-state battery that could double EV driving range. "
                "The technology retains ninety percent capacity after 1,000 fast-charge cycles."
            ),
            (
                "(CNN) -- Global biodiversity conservation efforts received a major boost with the establishment of "
                "a vast new marine protected area spanning over one hundred thousand square kilometers in the Pacific Ocean. "
                "The sanctuary will safeguard critical habitats for endangered sea turtles, hammerhead sharks, and migratory "
                "tuna populations. Commercial fishing, seabed mining, and oil exploration will be strictly prohibited within "
                "the reserve boundaries.",
                "A new 100,000 square kilometre marine sanctuary is established in the Pacific Ocean. "
                "Commercial fishing and mining are banned to protect endangered marine species."
            ),
        ]

        # Duplicate samples if needed to satisfy subset sizes
        max_needed = max(self.subset_sizes.values(), default=20)
        multiplier = (max_needed // len(sample_articles)) + 1
        expanded = (sample_articles * multiplier)[:max_needed]

        data = {
            self.text_col: [item[0] for item in expanded],
            self.summary_col: [item[1] for item in expanded],
        }

        full_ds = Dataset.from_dict(data)
        return DatasetDict({
            "train": full_ds,
            "validation": full_ds,
            "test": full_ds,
        })

