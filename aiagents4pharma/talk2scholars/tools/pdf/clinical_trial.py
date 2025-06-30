"""
LangGraph PDF Retrieval-Augmented Generation (RAG) Tool

This tool answers user questions by retrieving and ranking relevant text chunks from PDFs
and invoking an LLM to generate a concise, source-attributed response. It supports
single or multiple PDF sources—such as Zotero libraries, arXiv papers, or direct uploads.

Workflow:
  1. (Optional) Load PDFs from diverse sources into a FAISS vector store of embeddings.
  2. Rerank candidate papers using NVIDIA NIM semantic re-ranker.
  3. Retrieve top-K diverse text chunks via Maximal Marginal Relevance (MMR).
  4. Build a context-rich prompt combining retrieved chunks and the user question.
  5. Invoke the LLM to craft a clear answer with source citations.
  6. Return the answer in a ToolMessage and citations as artifact for LangGraph to dispatch.
"""

import logging
import os
import time
from typing import Annotated, Dict, Any
import hydra

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from .utils.retrieve_chunks import retrieve_relevant_chunks
from .utils.tool_helper import QAToolHelper

from .utils.vector_store import Vectorstore

# Set up logging with configurable level
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, log_level))

def load_hydra_config() -> Any:
    """
    Load the configuration using Hydra and return the configuration for the Q&A tool.
    """
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="config",
            overrides=["tools/clinical_trial=default"],
        )
        config = cfg.tools.clinical_trial
        logger.debug("Loaded Clinical Trial tool configuration.")
        return config

# Helper for managing state, vectorstore, reranking, and formatting
helper = QAToolHelper()
# Load configuration and start logging
config = load_hydra_config()


class ClinicalTrialInput(BaseModel):
    """
    Pydantic schema for the clinical trial tool input.

    Fields:
      tool_call_id: LangGraph-injected call identifier for tracking.
      state: Shared agent state dict containing:
        - article_data: metadata mapping of paper IDs to info (e.g., 'pdf_url', title).
        - text_embedding_model: embedding model instance for chunk indexing.
        - llm_model: chat/LLM instance for answer generation.
        - vector_store: optional pre-built Vectorstore for retrieval.
    """

    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[dict, InjectedState]


@tool(args_schema=ClinicalTrialInput, parse_docstring=True)
def trial_summarization(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """
    LangGraph tool for Retrieval-Augmented Generation over Clinical Trial PDFs.

    Given a user question, this tool applies the following pipeline:
      1. Validates that embedding and LLM models, plus article metadata, are in state.
      2. Initializes or reuses a FAISS-based Vectorstore for PDF embeddings.
      3. Loads one or more PDFs (from Zotero, arXiv, uploads) as text chunks into the store.
      4. Retrieves the most relevant and diverse text chunks via Maximal Marginal Relevance.
      5. Constructs an LLM prompt combining contextual chunks and the query.
      6. Invokes the LLM to generate an answer, appending source attributions.
      7. Returns a LangGraph Command with ToolMessage containing the answer and citations as artifact.

    Args:
      state (dict): Injected agent state; must include:
        - article_data: mapping paper IDs → metadata (pdf_url, title, etc.)
        - text_embedding_model: embedding model instance.
        - llm_model: chat/LLM instance.
      tool_call_id (str): Internal identifier for this tool invocation.

    Returns:
      Command[Any]: updates conversation state with a ToolMessage(answer) and citations as artifact.

    Raises:
      ValueError: when required models or metadata are missing in state.
      RuntimeError: when no relevant chunks can be retrieved for the query.
    """
    call_id = f"ct_call_{time.time()}"
    logger.info(
        "Starting Clinical Trial tool call %s",
        call_id,
    )
    helper.start_call(config, call_id)

    # Extract models and article metadata
    text_emb, llm_model, article_data = helper.get_state_models_and_data(state)
    # clinical trial data variable
    clinical_data: Dict[str, Any] = {}
    clinical_data["RMDSVIR"] = {
        "Title": "A Phase 3 Randomized, Double-Blind Placebo-Controlled Trial to Evaluate the Efficacy and Safety of Remdesivir (GS-5734™) Treatment of COVID-19 in an Outpatient Setting",
        "Sponsor": "Gilead Sciences, Inc",
        "Objective": """The purpose of this trial is to evaluate treatment with intravenous
                    (IV) administered remdesivir (RDV, GS-5734) in an outpatient
                    setting in participants with confirmed coronavirus disease 2019
                    (COVID-19) who are at risk for disease progression.
                    The primary objectives of this study are as follows:
                     To evaluate the efficacy of RDV in reducing the rate of
                    COVID-19 related hospitalization or all-cause death in
                    non-hospitalized participants with early stage COVID-19
                     To evaluate the safety of RDV administered in an outpatient
                    setting
                    The secondary objectives of this study are as follows:
                     To evaluate the efficacy of RDV in reducing the rate of
                    COVID-19 related medically attended visits (MAVs; medical
                    visits attended in person by the participant and a health care
                    professional) or all-cause death in non-hospitalized participants
                    with early stage COVID-19
                     To determine the antiviral activity of RDV on severe acute
                    respiratory syndrome (SARS)-coronavirus (CoV)-2 viral load
                     To assess the impact of RDV on symptom duration and severity""",
        "Publication Date": "14 January 2021",
        "URL": "https://cdn.clinicaltrials.gov/large-docs/52/NCT04501952/Prot_000.pdf",
        "pdf_url": "https://cdn.clinicaltrials.gov/large-docs/52/NCT04501952/Prot_000.pdf",
        "filename": "RMDSVIR.pdf",
        "source": "clinical_trial",
        "cl_id": "RMDSVIR",
    }
    clinical_data["RMDSVIR2"]= {
        "Title": "Remdesivir for the Treatment of Covid-19 — Final Report",
        "Sponsor": "J.H. Beigel, K.M. Tomashek, L.E. Dodd",
        "Objective": """We conducted a double-blind, randomized, placebo-controlled trial of 
        intravenous remdesivir in adults who were hospitalized with Covid-19 and had 
        evidence of lower respiratory tract infection. Patients were randomly assigned 
        to receive either remdesivir (200 mg loading dose on day 1, followed by 100 mg 
        daily for up to 9 additional days) or placebo for up to 10 days. The primary 
        outcome was the time to recovery, defined by either discharge from the hospital 
        or hospitalization for infection-control purposes only.""",
        "Publication Date": "8 June 2019",
        "URL": "https://europepmc.org/articles/PMC8406992?pdf=render",
        "pdf_url": "https://europepmc.org/articles/PMC8406992?pdf=render",
        "filename": "RMDSVIR2.pdf",
        "source": "clinical_trial",
        "cl_id": "RMDSVIR2",
    }

    # Initialize or reuse vector store, then load candidate papers
    vs = helper.init_vector_store(text_emb)
    candidate_ids = list(clinical_data.keys())
    print(f"Candidate paper IDs: {candidate_ids}")
    logger.info("%s: Candidate paper IDs for reranking: %s", call_id, candidate_ids)
    helper.load_candidate_papers(vs, clinical_data, candidate_ids)
    #building a vector store
    vs.build_vector_store()
    questions = [
      "What are the Patient demographics and sample sizes in the article?",
       "What were the Dosages, Dosing regimens and administration schedules?",
       "What are the inclusion and exclusion criteria?"
       "What are the Indications and therapeutic areas?",
       "What are the Outcome measures (e.g., IBDQ, CDAI, CRP levels).",
    ]
    response_text = ""
    sources = set()
    for question in questions:
        try:
          logger.info(f"Question: {question}")
          time.sleep(3)
          # Rerank papers and retrieve top chunks
          selected_ids = helper.run_reranker(vs, question, candidate_ids)

          relevant_chunks = retrieve_relevant_chunks(
              vs, query=question, paper_ids=selected_ids, top_k=config.top_k_chunks
          )
          if not relevant_chunks:
              msg = f"No relevant chunks found for question: '{question}'"
              logger.warning("%s: %s", call_id, msg)
              raise RuntimeError(msg)

          # Generate answer and format with sources
          response_fa = helper.format_answer(
              question, relevant_chunks, llm_model, clinical_data,config=config
          )
          response_text += response_fa['answer']
          for cits in response_fa['citations']:
              sources.add(cits)
        except Exception as e:
            print(f"Error processing question: {question} - {e}")
    print(f"Response text: {response_text}")

    content = f"{response_text}. Citations to be displayed are sent as an artifact."
    srcs = "\nSources:\n\n"
    for s in sources:
        srcs += f"{s}\n"
    print(f"Sources: {srcs}")
    logger.info("Sending back generated response and citations as an artifact")
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=content,
                    artifact=srcs,
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )
