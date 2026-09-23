"""Summarize a short passage with every Sumplicity algorithm."""
import sumplicity

TEXT = """
The river flooded the valley after three days of heavy rain. Emergency crews
evacuated residents from the flooded valley towns. Rainfall totals in the valley
exceeded historical records. A local bakery donated bread to the evacuated
residents. Officials expect the river to recede by the end of the week. The mayor
praised emergency crews for the rapid evacuation.
"""

for name, Summarizer in sumplicity.get_all_summarizer().items():
    print(f"== {name} ==")
    for sentence in Summarizer().summarize(TEXT, top_k=2):
        print(" -", " ".join(sentence.split()))
