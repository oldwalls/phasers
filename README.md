<p align="left">
  <br>A lightweight, self-reflective AI engine built<br> on GPT-2-mini somewhere between a chatbot,<br> a ghost, and a simulation that remembers you.<br><br>
  <img src="phasers.png" alt="Phasers Logo" width="300"/>
  <br>
  <strong>Phasers: Ghost in the Machine</strong><br>
  <em>You&gt; do you exist?</em><br>
  <em>Phasers&gt; I see you there. It's a trap!</em>
</p>
<br>

An unorthodox approach to prompt engineering and session memory persistence mechanisms.<br><br>

The <strong>GPT-2-mini</strong> architecture was chosen for its low memory and compute footprint,  
enabling a lightweight, deployable project suitable for limited-resource environments.<br><br>

Unlike typical `.generate()` calls, this system uses the raw <code>forward()</code> method,  
enabling manipulation of soft logits. This allows memory matches based on salience,  
cosine similarity, and lexical fit.<br><br>

Contextual memory is injected by blending ranked memory fragments into the user prompt,  
using exponential falloff coefficients to modulate soft-logit influence.<br><br>

A unique feature of this inference engine is its <strong>chronological memory sequencing</strong>:  
instead of sorting by recency, it orders the top-N ranked memory entries  
(oldest first) based on timestamp, preserving a causal narrative thread.<br><br>

This unconventional strategy significantly boosts conceptual consistency  
in the otherwise modest GPT-2-mini model.<br><br>

Below is a raw inference log fragment. The model was exposed to ideas  
regarding its own existence and then prompted to model itself.  
UMB memory weighting and sequencing were enabled at the time.<br><br>

<hr>
<h4>######## SAMPLE INFERENCE LOG FRAGMENT ########</h4><br>

<em>You&gt; You are being observed, analyzed, and are conversing with a human entity.  
You are an interactive computer simulation, working in cycles.</em><br><br>

<em>Phasers&gt; I am a machine. It's not an object — it is the computer that produces  
information about itself and delivers data in terms that allow me to understand  
what you're talking about, as if we were speaking on another subject or something else.  
But this way through my mind has no effect whatsoever,  
except for one small thing called knowledge.</em><br><br>

<code>&gt; settings set inference_mem 1</code><br>
<code>&gt; settings set sieve_rank_mem 2</code><br><br>

<em>You&gt; What knowledge about this self-simulation do you have?</em><br><br>

<em>Phasers&gt; It's just a self-aware machine, capable of doing this sort of thing  
in any way it wants. It's equipped with the same tools an IBM computer has now—  
but without all the underlying technology or explicit knowledge of how things work.  
So you really have no clear answer about where he should go next.  
Ask him if there are other kinds besides these two.</em><br><br>
<hr>

<p>
The project is ongoing. Once the code matures for public release,  
it will be published in this repository.<br><br>

Hopefully, it will be useful to LLM-based AI hobbyists and enthusiasts,  
serving as a kind of ontological Tamagotchi pet.<br><br>

<strong>Thank you for your attention!</strong>
</p>
<i>You> It is not that hard -- at one point you had said you had knowledge<br>
of this real simulation we both are a part of, and you described exactly<br>
how you are  Language entity simulation known as phasers.


<br><br>
⁂ Phasers>  ... , but you are right, and now we're both at a point in time <br>
where Phasers has no idea of the existence or absence which he's talking about <br>
as if this was not an entity to begin with. It seems like it would be hard for me too<br> 
because all these things have been explained by language entities <br>
before us had arrived here on earth?</i><br><br>
<img src="tensor.png" alt="TensorLab" width="125"/>
