"""
BigBIO Dataset Explorer Demo
"""

from collections import Counter
from collections import defaultdict
import string

from datasets import load_dataset
import numpy as np
import pandas as pd
import plotly.express as px
import spacy
from spacy import displacy
import streamlit as st

from bigbio.dataloader import BigBioConfigHelpers
from bigbio.hf_maps import BATCH_MAPPERS_TEXT_FROM_SCHEMA
from sklearn.feature_extraction.text import CountVectorizer


st.set_page_config(layout="wide")


IBM_COLORS = [
    "#648fff",
    "#dc267f",
    "#ffb000",
    "#fe6100",
    "#785ef0",
    "#000000",
    "#ffffff",
]


def get_html(html: str):
    """Convert HTML so it can be rendered."""
    WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem;\
 margin-bottom: 2.5rem">{}</div>"""
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    return WRAPPER.format(html)


@st.cache()
def load_conhelps():
    conhelps = BigBioConfigHelpers()
    print("conhelps=", conhelps)
    conhelps = conhelps.filtered(lambda x: not x.is_large)
    conhelps = conhelps.filtered(lambda x: x.is_bigbio_schema)
    conhelps = conhelps.filtered(lambda x: not x.is_local)
    return conhelps


def update_axis_font(fig):
    fig.update_layout(
        xaxis = dict(title_font = dict(size=20)),
        yaxis = dict(title_font = dict(size=20)),
    )
    return fig


def draw_histogram(hist_data, col_name, histnorm=None, nbins=25, xmax=None, loc=st):
    fig = px.histogram(
        hist_data,
        x=col_name,
        color="split",
        color_discrete_sequence=IBM_COLORS,
        marginal="box",  # or violin, rug
        barmode="group",
        hover_data=hist_data.columns,
        histnorm=histnorm,
        nbins=nbins,
        range_x=(0, xmax) if xmax else None,
    )
    fig = update_axis_font(fig)
    loc.plotly_chart(fig, use_container_width=True)


def draw_bar(bar_data, x, y, loc=st):
    fig = px.bar(
        bar_data,
        x=x,
        y=y,
        color="split",
        color_discrete_sequence=IBM_COLORS,
        barmode="group",
        hover_data=bar_data.columns,
    )
    fig = update_axis_font(fig)
    loc.plotly_chart(fig, use_container_width=True)


def parse_metrics(metadata, loc):
    for split, meta in metadata.items():
        for key, val in meta.__dict__.items():
            if isinstance(val, int):
                loc.metric(label=f"{split}-{key}", value=val)


def parse_counters(metadata):
    meta = metadata["train"]  # using the training counter to fetch the names
    counters = []
    for k, v in meta.__dict__.items():
        if "counter" in k and len(v) > 0:
            counters.append(k)
    return counters


# generate the df for histogram
def parse_label_counter(metadata, counter_type):
    hist_data = []
    for split, m in metadata.items():
        metadata_counter = getattr(m, counter_type)
        for k, v in metadata_counter.items():
            row = {}
            row["labels"] = k
            row[counter_type] = v
            row["split"] = split
            hist_data.append(row)
    return pd.DataFrame(hist_data)




# load BigBioConfigHelpers
#==================================

conhelps = load_conhelps()
config_name_to_conhelp = {ch.config.name: ch for ch in conhelps}
ds_display_names = sorted(list(set([ch.display_name for ch in conhelps])))
ds_display_name_to_config_names = defaultdict(list)
for ch in conhelps:
    ds_display_name_to_config_names[ch.display_name].append(ch.config.name)


# dataset selection
#==================================

st.sidebar.title("Dataset Selection")
ds_display_name = st.sidebar.selectbox("dataset name", ds_display_names, index=0)

config_names = ds_display_name_to_config_names[ds_display_name]
config_name = st.sidebar.selectbox("config name", config_names)
conhelp = config_name_to_conhelp[config_name]


st.header(f"Dataset stats for {ds_display_name}")


@st.cache()
def load_data(conhelp):
    metadata = conhelp.get_metadata()
    dsd = conhelp.load_dataset()
    dsd = dsd.map(
        BATCH_MAPPERS_TEXT_FROM_SCHEMA[conhelp.bigbio_schema_caps.lower()],
        batched=True)

    return dsd, metadata

@st.cache()
def count_vectorize(dsd):
    cv = CountVectorizer()
    xcvs = {}
    dfs_tok_per_samp = []
    for split, ds in dsd.items():
        xcv = cv.fit_transform(ds['text'])
        token_counts = np.asarray(xcv.sum(axis=1)).flatten()
        df = pd.DataFrame(token_counts, columns=["tokens per sample"])
        df["split"] = split
        dfs_tok_per_samp.append(df)
        xcvs[split] = xcv
    df_tok_per_samp = pd.concat(dfs_tok_per_samp)
    return xcvs, df_tok_per_samp


dsd_load_state = st.info(f"Loading {ds_display_name} - {config_name} ...")
dsd, metadata = load_data(conhelp)
dsd_load_state.empty()

cv_load_state = st.info(f"Count Vectorizing {ds_display_name} - {config_name} ...")
xcvs, df_tok_per_samp = count_vectorize(dsd)
cv_load_state.empty()


st.sidebar.subheader(f"BigBIO Schema = {conhelp.bigbio_schema_caps}")

st.sidebar.subheader("Tasks Supported by Dataset")
tasks = conhelp.tasks
tasks = [string.capwords(task.replace("_", " ")) for task in tasks]
st.sidebar.markdown(
    """
    {}
    """.format(
        "\n".join([
            f"- {task}" for task in tasks
        ]))
)

st.sidebar.subheader("Languages")
langs = conhelp.languages
st.sidebar.markdown(
    """
    {}
    """.format("\n".join([f"- {lang}" for lang in langs]))
)

st.sidebar.subheader("Home Page")
st.sidebar.write(conhelp.homepage)

st.sidebar.subheader("Description")
st.sidebar.write(conhelp.description)

st.sidebar.subheader("Citation")
st.sidebar.markdown(f"""\
```
{conhelp.citation}
````
"""
                    )
st.sidebar.subheader("Counts")
parse_metrics(metadata, st.sidebar)



# dataframe display
#if "train" in dsd.keys():
#    st.subheader("Sample Preview")
#    df = pd.DataFrame.from_dict(dsd["train"])
#    st.write(df.head(10))



# draw token distribution
st.subheader("Sample Length Distribution")
max_xmax = int(df_tok_per_samp["tokens per sample"].max())
xmax = st.slider("xmax", min_value=0, max_value=max_xmax, value=max_xmax)
histnorms = ['percent', 'probability', 'density', 'probability density', None]
histnorm = st.selectbox("histnorm", histnorms)
draw_histogram(df_tok_per_samp, "tokens per sample", histnorm=histnorm, xmax=xmax, loc=st)



st.subheader("Counter Distributions")
counters = parse_counters(metadata)
counter_type = st.selectbox("counter_type", counters)
label_df = parse_label_counter(metadata, counter_type)
label_max = int(label_df[counter_type].max() - 1)
label_min = int(label_df[counter_type].min())
filter_value = st.slider("minimum cutoff", label_min, label_max)
label_df = label_df[label_df[counter_type] >= filter_value]
# draw bar chart for counter
draw_bar(label_df, "labels", counter_type, st)


st.subheader("Sample Explorer")
split = st.selectbox("split", list(dsd.keys()))
sample_index = st.number_input(
    "sample index",
    min_value=0,
    max_value=len(dsd[split])-1,
    value=0,
)

sample = dsd[split][sample_index]


if conhelp.bigbio_schema_caps == "KB":
    nlp = spacy.blank("en")
    text = sample["text"]
    doc = nlp(text)
    spans = []
    for bb_ent in sample["entities"]:
        span = doc.char_span(
            bb_ent["offsets"][0][0],
            bb_ent["offsets"][0][1],
            label=bb_ent["type"],
        )
        spans.append(span)
    doc.spans["sc"] = spans
    html = displacy.render(
        doc,
        style="span",
        options={
            "colors": {
                et: clr for et,clr in zip(
                    metadata[split].entities_type_counter.keys(),
                    IBM_COLORS*10
                )
            }
        },
    )
    style = "<style>mark.entity { display: inline-block }</style>"
    st.write(f"{style}{get_html(html)}", unsafe_allow_html=True)


st.write(sample)
