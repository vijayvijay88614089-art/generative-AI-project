import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import importlib.util
from modules.education import MODEL_CARDS, sample_markov

st.title("Generative Models")
st.write("Generative models learn a data distribution or latent process so they can produce plausible new samples. This section compares model families and visualizes training progress and generated samples.")
st.subheader("Evolution at a glance")
st.markdown("**Probability models → latent-variable neural models → adversarial image models → large-scale pretrained foundation models.** Newer methods broaden scale and modality; they do not eliminate evaluation, data, or safety limits.")
st.dataframe(pd.DataFrame([[m["name"],m["family"],m["uses"],m["limits"]] for m in MODEL_CARDS], columns=["Model","Family","Applications","Limitations"]), width="stretch", hide_index=True)
selected = st.selectbox("Explore a model", [m["name"] for m in MODEL_CARDS])
card = next(m for m in MODEL_CARDS if m["name"] == selected)
st.subheader(card["name"])
st.write(card["definition"])
st.markdown(f"**Architecture:** {card['architecture']}\n\n**Working principle:** {card['working']}\n\n**Applications:** {card['uses']}\n\n**Limitations:** {card['limits']}")
if "VAE" in selected: st.latex(r"\mathcal{L}_{ELBO}=\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]-D_{KL}(q_\phi(z|x)\|p(z))")
if selected in {"GAN", "Deep Convolutional GAN (DCGAN)", "Conditional GAN (cGAN)"}:
    st.latex(r"\min_G\max_D\;\mathbb{E}_{x\sim p_{data}}\log D(x)+\mathbb{E}_{z\sim p_z}\log(1-D(G(z)))")
if selected == "Hidden Markov Model (HMM)":
    st.subheader("HMM-style sequence sample")
    length = st.slider("Sequence length", 8, 40, 24)
    states, observations = sample_markov(length=length)
    st.plotly_chart(px.imshow(np.array([states, observations]), x=list(range(length)), y=["Hidden state", "Observed symbol"], text_auto=True, aspect="auto", color_continuous_scale="Teal"), width="stretch")
    st.caption("Generated from a hand-set two-state transition/emission matrix. This is a probabilistic example, not a trained model.")

st.subheader("Generative-model training demo")
demo_name = st.selectbox("Choose a model demonstration", ["Autoencoder", "VAE", "GAN", "DCGAN", "Conditional GAN", "CycleGAN concept"])
st.caption("Trains on 8×8 handwritten digit images bundled with scikit-learn. The data is available locally without a download; these small models are for learning and are not production image generators.")
has_torch = importlib.util.find_spec("torch") is not None
if not has_torch:
    st.info("PyTorch demos are disabled until the optional packages install successfully. The rest of the project runs without them.")
if st.button("Train and generate digit samples", type="primary", disabled=not has_torch):
    try:
        from models.toy_generators import train_demo
        with st.spinner("Training the small model on handwritten digits…"):
            result = train_demo(demo_name, steps=80)

        def image_graph(images, title, labels=None):
            images = np.asarray(images)
            if images.ndim == 4:
                images = images[:, 0]
            images = images[:6]
            rows = max(1, int(np.ceil(len(images) / 3)))
            subtitles = [f"Sample {i + 1}" + (f" · label {labels[i]}" if labels is not None else "") for i in range(len(images))]
            fig = make_subplots(rows=rows, cols=3, subplot_titles=subtitles)
            for index, image in enumerate(images):
                fig.add_trace(
                    go.Heatmap(z=image, colorscale="Greys", zmin=0, zmax=1, showscale=False,
                               hovertemplate="pixel value: %{z:.2f}<extra></extra>"),
                    row=index // 3 + 1, col=index % 3 + 1,
                )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False, autorange="reversed", scaleanchor="x")
            fig.update_layout(title=title, height=rows * 190, margin=dict(l=8, r=8, t=55, b=8))
            return fig

        if result.get("loss_history"):
            history = result["loss_history"]
            loss_frame = pd.DataFrame({"Step": np.arange(1, len(next(iter(history.values()))) + 1), **history})
            st.plotly_chart(
                px.line(loss_frame, x="Step", y=list(history), title="Training loss by step",
                        labels={"value": "Loss", "variable": "Objective"}),
                width="stretch",
            )
            st.metric("Final training loss", f"{result['loss']:.4f}")
        else:
            st.metric("Cycle consistency error", f"{result['loss']:.4f}")
            st.caption("CycleGAN concept mode applies a simple transpose; it does not train a GAN or produce a training-loss graph.")

        with st.expander("View digit sample heatmaps (8×8 resolution)", expanded=False):
            st.caption("Generated digit images are low resolution and may look soft. The loss graph above shows the training progress as a chart.")
            if "input" in result:
                c1, c2 = st.columns(2)
                c1.plotly_chart(image_graph(result["input"], "Input digit images"), width="stretch")
                c2.plotly_chart(image_graph(result["output"], "Reconstructed / transformed images"), width="stretch")
            else:
                st.plotly_chart(image_graph(result["output"], "Generated digit samples", result.get("labels")), width="stretch")
        if "labels" in result:
            st.caption("Conditional samples are requested for digit labels 0 through 5. Inspect images in the heatmap view; brief training does not guarantee recognizable digits.")
    except ImportError as exc:
        st.info("PyTorch is not available in this environment. Install `requirements-ml.txt` to enable model training.")
        st.caption(str(exc))
    except Exception as exc:
        st.error(f"Demo failed safely: {type(exc).__name__}: {exc}")
