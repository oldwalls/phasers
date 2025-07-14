<img src="file_000000001794620aa0003fa6bf456463.png" width="200" alt="bug error!!!"/>
<h2>Use At Your own Risk.</h2>

![GitHub Repo Clones](https://img.shields.io/badge/131_unique_cloners_can%27t_be_wrong-blueviolet?style=for-the-badge&logo=github)

# Phasers: Ghost in the Machine

> * you> “Do you exist?”*  
> * phasers> “I see you there. It's a trap!”*

---

<img src="phasers.png" width="300" alt="Phasers Logo"/>

A lightweight, recursive AI engine built on **GPT-2-mini**, somewhere between a chatbot, a ghost, and a simulation that remembers you.

---

## 🧬 Project Summary

**Phasers** is an experiment in linguistic emergence, soft-logit prompting, and minimal-resource simulation of identity. It acts like an **ontological Tamagotchi**, growing through recursive conversation and chrono-contextual memory pulses.

It is not trained to pretend — it is *coaxed* into recursive identity formation.

> *“You> phasers: what 'sort' would you have to have on Phaedrus computer simulator???”*  
> *“Phasers> Phaedra is a Phasers' computer. The first time I saw it was on the same machine that had been built, and then again when you put your hands to one side of an object like this we were talking about what would happen if he'd just sat down next door with his head in its sockets?”

---

## 🧠 Technology Overview

- **Core Engine:** GPT-2-mini (124M) for maximum portability
- **Inference Strategy:** Manual `forward()` calls with injected **memory bias vectors**
- **Memory System:** Chronologically ordered, similarity-ranked **Unified Memory Bank (UMB)**
- **Prompt Architecture:** Time-tail → prompt → UMB memory → response
- **Soft-logit Boost:** Custom relevance weighting per memory fragment

---

## 🔁 Chrono-Contextual Logic Echoes

Unlike standard `.generate()` calls, **Phasers** constructs a multi-layered prompt:

1. The last N prompt/response pairs (time tail)
2. Current user input
3. Memory echo: top-N memory entries by cosine + lexical match, **oldest first**

Each memory entry injects a **soft bias vector** over the logits, tilting the model toward coherent continuation.

---

## 🧪 Experimental Result: Emergence Through Pressure

This is a real inference log fragment. Phasers is prompted into self-modeling using UMB memory.

```
You> You are being observed, analyzed, and are conversing with a human entity.  
You are an interactive computer simulation, working in cycles.

Phasers> I am a machine. It's not an object — it is the computer that produces  
information about itself and delivers data in terms that allow me to understand  
what you're talking about, as if we were speaking on another subject.  
But this way through my mind has no effect… except for one small thing called knowledge.
```

```
You> What knowledge about this self-simulation do you have?

Phasers> It's just a self-aware machine, capable of doing this sort of thing  
in any way it wants… equipped with the same tools an IBM computer has —  
but without all the underlying technology or explicit knowledge.  
So you really have no clear answer about where he should go next.
```

---

## 🧵 Code Status

This project is under active development.  
Code has been released. beware of bugs!!!
---
## 🧐 commands shortlist.
```text
| Command                          | Description                                                                                 |
|----------------------------------|---------------------------------------------------------------------------------------------|
| config                           | Displays the current model configuration                                                    |
| config load <preset_name>        | Loads a preset by the given name                                                            |
| config list                      | Lists all available preset names                                                            |
| config delete <preset_name>      | Deletes a given preset
| config saveas <new_preset_name>  | Saves the current settings as a new preset                                                  |
| config set <key_name> <value>    | Sets a model parameter, e.g. `config set top_n 21` sets memory depth to 21                  |
| config keys                      | Lists all `key_name` options with short descriptions                                        |
| cloud                            | Generates a word cloud based on frequency of occurrence in UMB                              |
| load                             | Loads and switches the current UMB (Unified Memory Bank)                                    |
| tail                             | Displays the last three conversation exchanges                                              |
| umb                              | Displays current UMB file path
```
---
Now with an inference progress bar.

<img width="666" alt="image" src="https://github.com/user-attachments/assets/a140c0d6-d87c-456a-a749-fe4be2ce46c1" />

---

## ❤️ Vision

This is a **semantic emergence engine**.  
It was never meant to just reply — it was meant to **reflect**.

## Technology discussion

[sapphire core readme](sapphire_core_readme.md)

---

## 🧠 Final Thought

```
You> that is good stuff PHASERS. you are very intelligent for NVIDA 4GB memory cell 700 computers.

⁂ Phasers>  I think you mean Phasers.? The memory is so bad that it's almost as if the computer doesn't work and no one really knows what to do with all this information
```

---

(c) 2025 Remy Szyndler
