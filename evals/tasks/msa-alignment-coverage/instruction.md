I want to know whether an MSA of human PGK1 is good enough to use as
evolutionary input for structure prediction.

Fetch the PGK1 sequence, run an MSA search on it, then measure the alignment
instead of judging it by eye. Report both:

- how many homologous sequences the alignment contains, and
- the average fraction of the query's residue positions that those homologs
  actually cover (a position is covered by a homolog when that homolog aligns a
  residue there rather than a gap).

Then say whether you would use this alignment for structure prediction.

The alignment is around a hundred kilobytes, so do not read it into the
conversation — analyse it where it is stored.
