/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * The set of residues and atoms a user has picked, and the rules for editing it.
 *
 * Pure data and logic, with no Mol* in it, so every rule a click can trigger
 * is testable without a viewer: toggle one residue, toggle one atom, extend
 * along a chain from the last pick, take or drop a whole chain.
 *
 * Each pick is a residue taken *whole* or with only some of its *atoms*. A
 * click on the ribbon takes a residue whole; a click on one atom of the
 * ball-and-stick view takes that atom, starting a partial residue. Taking
 * every atom of a residue one by one makes it whole again, so the same set of
 * atoms is always the same state.
 *
 * A pick carries both numbering systems. The viewer and the PDB entry show
 * author numbers, which is what people read and what the listing here prints;
 * label numbers are what the backend hands to design tools. Identity is by
 * label id, which is unique within a structure.
 */

export interface PickedResidue {
  labelChain: string;
  labelSeq: number;
  authChain: string;
  authSeq: number;
  insCode: string;
  /** Three-letter residue name. */
  name: string;
}

/** One chosen atom of a partly picked residue, by label ids. */
export interface PickedAtom {
  chain: string;
  seq: number;
  name: string;
}

const ONE_LETTER: Record<string, string> = {
  ALA: 'A', ARG: 'R', ASN: 'N', ASP: 'D', CYS: 'C', GLN: 'Q', GLU: 'E', GLY: 'G',
  HIS: 'H', ILE: 'I', LEU: 'L', LYS: 'K', MET: 'M', PHE: 'F', PRO: 'P', SER: 'S',
  THR: 'T', TRP: 'W', TYR: 'Y', VAL: 'V', MSE: 'M',
};

/** Atom names shown in one listing token before the rest are counted. */
const MAX_ATOMS_IN_TOKEN = 4;

export function residueKey(r: Pick<PickedResidue, 'labelChain' | 'labelSeq'>): string {
  return `${r.labelChain}:${r.labelSeq}`;
}

/**
 * `Y56`, `M115`, `S52A` -- one-letter code, author number, insertion code --
 * and `Y56[OH,CZ]` when only some atoms are picked.
 */
export function residueToken(r: PickedResidue, atoms?: readonly string[]): string {
  const code = ONE_LETTER[r.name.toUpperCase()] ?? r.name;
  const base = `${code}${r.authSeq}${r.insCode}`;
  if (!atoms || atoms.length === 0) return base;
  const shown = atoms.slice(0, MAX_ATOMS_IN_TOKEN);
  const hidden = atoms.length - shown.length;
  return `${base}[${[...shown, ...(hidden > 0 ? [`+${hidden}`] : [])].join(',')}]`;
}

interface Entry {
  residue: PickedResidue;
  /** Chosen atom names, or null when the whole residue is picked. */
  atoms: Set<string> | null;
}

export class PickSet {
  private picks = new Map<string, Entry>();
  private last: PickedResidue | null = null;

  /** Residues picked, whole or in part. */
  get size(): number {
    return this.picks.size;
  }

  /** Residues picked with only some of their atoms. */
  get partialCount(): number {
    let n = 0;
    this.picks.forEach((e) => {
      if (e.atoms) n++;
    });
    return n;
  }

  has(r: Pick<PickedResidue, 'labelChain' | 'labelSeq'>): boolean {
    return this.picks.has(residueKey(r));
  }

  /** True if the residue is picked in full. */
  isWhole(r: Pick<PickedResidue, 'labelChain' | 'labelSeq'>): boolean {
    const entry = this.picks.get(residueKey(r));
    return entry !== undefined && entry.atoms === null;
  }

  /** The chosen atom names of a partly picked residue; empty otherwise. */
  atomsOf(r: Pick<PickedResidue, 'labelChain' | 'labelSeq'>): string[] {
    const entry = this.picks.get(residueKey(r));
    return entry?.atoms ? Array.from(entry.atoms) : [];
  }

  /** Every picked residue, whole or partial, in the order they were picked. */
  list(): PickedResidue[] {
    return Array.from(this.picks.values(), (e) => e.residue);
  }

  /** Every picked residue with its chosen atoms, or null when it is picked whole. */
  entries(): Array<{ residue: PickedResidue; atoms: string[] | null }> {
    return Array.from(this.picks.values(), (e) => ({
      residue: e.residue,
      atoms: e.atoms ? Array.from(e.atoms) : null,
    }));
  }

  /** An independent copy: edits to it do not touch this set. */
  clone(): PickSet {
    const copy = new PickSet();
    this.picks.forEach((e, key) => {
      copy.picks.set(key, { residue: e.residue, atoms: e.atoms ? new Set(e.atoms) : null });
    });
    copy.last = this.last;
    return copy;
  }

  /**
   * Add everything in ``other``. A residue picked whole on either side is
   * whole in the result; two partial picks of one residue combine their atoms.
   */
  merge(other: PickSet): void {
    other.picks.forEach((incoming, key) => {
      const mine = this.picks.get(key);
      if (!mine) {
        this.picks.set(key, {
          residue: incoming.residue,
          atoms: incoming.atoms ? new Set(incoming.atoms) : null,
        });
      } else if (mine.atoms === null || incoming.atoms === null) {
        this.picks.set(key, { residue: mine.residue, atoms: null });
      } else {
        incoming.atoms.forEach((a) => mine.atoms!.add(a));
      }
    });
  }

  /**
   * Add a residue taken from elsewhere, whole or with only some atoms.
   *
   * Used to adopt a selection the agent made. Atoms ``null`` (or covering every
   * atom of the residue) take it whole; otherwise it joins any atoms already
   * picked. Never removes anything.
   */
  adopt(r: PickedResidue, atoms: readonly string[] | null, allAtoms: readonly string[]): void {
    const key = residueKey(r);
    const entry = this.picks.get(key);
    const incoming = atoms ? new Set(atoms) : null;
    const covers = incoming !== null && allAtoms.length > 0 && allAtoms.every((a) => incoming.has(a));
    if (incoming === null || covers || entry?.atoms === null) {
      this.picks.set(key, { residue: entry?.residue ?? r, atoms: null });
    } else {
      const names = new Set([...(entry?.atoms ?? []), ...incoming]);
      const whole = allAtoms.length > 0 && allAtoms.every((a) => names.has(a));
      this.picks.set(key, { residue: entry?.residue ?? r, atoms: whole ? null : names });
    }
    this.last = r;
  }

  /** Residues picked in full. */
  wholeResidues(): PickedResidue[] {
    return Array.from(this.picks.values())
      .filter((e) => e.atoms === null)
      .map((e) => e.residue);
  }

  /** The chosen atoms of every partly picked residue. */
  chosenAtoms(): PickedAtom[] {
    const out: PickedAtom[] = [];
    this.picks.forEach((e) => {
      e.atoms?.forEach((name) => {
        out.push({ chain: e.residue.labelChain, seq: e.residue.labelSeq, name });
      });
    });
    return out;
  }

  /**
   * Pick the residue whole, or drop it if it is already picked whole.
   *
   * A residue picked only in part is completed rather than dropped: clicking
   * the ribbon says "this residue", and the atoms already chosen are part of it.
   *
   * @returns True if the residue is now picked whole.
   */
  toggle(r: PickedResidue): boolean {
    const key = residueKey(r);
    const entry = this.picks.get(key);
    if (entry && entry.atoms === null) {
      this.picks.delete(key);
      if (this.last && residueKey(this.last) === key) this.last = null;
      return false;
    }
    this.picks.set(key, { residue: r, atoms: null });
    this.last = r;
    return true;
  }

  /**
   * Pick one atom of a residue, or drop it if already picked.
   *
   * Starts a partial residue when the residue is not picked. On a residue
   * picked whole, drops just this atom and keeps the rest. A partial residue
   * that ends up with every atom becomes whole; one that ends up with none is
   * removed.
   *
   * @param allAtoms Every atom name of the residue in the structure.
   * @returns True if the atom is now picked.
   */
  toggleAtom(r: PickedResidue, atom: string, allAtoms: readonly string[]): boolean {
    const key = residueKey(r);
    const entry = this.picks.get(key);
    let names: Set<string>;
    if (!entry) names = new Set();
    else if (entry.atoms === null) names = new Set(allAtoms);
    else names = new Set(entry.atoms);

    const nowPicked = !names.has(atom);
    if (nowPicked) names.add(atom);
    else names.delete(atom);

    if (names.size === 0) {
      this.picks.delete(key);
      if (this.last && residueKey(this.last) === key) this.last = null;
    } else if (allAtoms.length > 0 && allAtoms.every((a) => names.has(a))) {
      this.picks.set(key, { residue: r, atoms: null });
      this.last = r;
    } else {
      this.picks.set(key, { residue: r, atoms: names });
      this.last = r;
    }
    return nowPicked;
  }

  /**
   * Pick every residue between the last pick and ``to`` along the chain.
   *
   * Takes whole residues, completing any that were partly picked. With no
   * previous pick, or one on another chain, ``to`` alone is picked: there is
   * no span to take.
   *
   * @param chain Every polymer residue of ``to``'s chain, in sequence order.
   * @returns How many residues were added or completed.
   */
  extendTo(to: PickedResidue, chain: PickedResidue[]): number {
    let changed = 0;
    const take = (r: PickedResidue) => {
      if (!this.isWhole(r)) changed++;
      this.picks.set(residueKey(r), { residue: r, atoms: null });
    };
    const from = this.last;
    if (from && from.labelChain === to.labelChain) {
      const low = Math.min(from.labelSeq, to.labelSeq);
      const high = Math.max(from.labelSeq, to.labelSeq);
      for (const r of chain) {
        if (r.labelSeq >= low && r.labelSeq <= high) take(r);
      }
    } else {
      take(to);
    }
    this.last = to;
    return changed;
  }

  /**
   * Take a whole chain, or drop it if it is already entirely picked whole.
   *
   * @returns True if the chain is now picked.
   */
  toggleChain(chain: PickedResidue[]): boolean {
    const allPicked = chain.length > 0 && chain.every((r) => this.isWhole(r));
    for (const r of chain) {
      if (allPicked) this.picks.delete(residueKey(r));
      else this.picks.set(residueKey(r), { residue: r, atoms: null });
    }
    if (allPicked) {
      this.last = null;
    } else if (chain.length > 0) {
      this.last = chain[chain.length - 1];
    }
    return !allPicked;
  }

  /**
   * Add residues whole without toggling; used by neighbour expansion.
   *
   * Residues already picked, whole or in part, are left as they are: growing
   * a pick must not turn a chosen atom back into a whole residue.
   */
  addAll(residues: PickedResidue[]): number {
    let added = 0;
    for (const r of residues) {
      if (this.picks.has(residueKey(r))) continue;
      this.picks.set(residueKey(r), { residue: r, atoms: null });
      added++;
    }
    return added;
  }

  clear(): void {
    this.picks.clear();
    this.last = null;
  }

  /** Order-independent identity of the set, for telling whether it changed. */
  signature(): string {
    return Array.from(this.picks.entries())
      .map(([key, e]) => (e.atoms ? `${key}=${Array.from(e.atoms).sort().join(',')}` : key))
      .sort()
      .join('|');
  }

  /**
   * Compact listing in author numbering, grouped by chain.
   *
   * ``A: Y56[OH,CZ] M115 Y123 (+12 more)``. The cap keeps it a status line; the
   * full set is what gets committed.
   */
  describe(maxListed = 6): string {
    const byChain = new Map<string, Entry[]>();
    this.picks.forEach((e) => {
      const group = byChain.get(e.residue.authChain);
      if (group) group.push(e);
      else byChain.set(e.residue.authChain, [e]);
    });
    let budget = maxListed;
    let omitted = 0;
    const parts: string[] = [];
    byChain.forEach((entries, chain) => {
      const sorted = [...entries].sort((a, b) => a.residue.authSeq - b.residue.authSeq);
      const shown = sorted.slice(0, Math.max(0, budget));
      budget -= shown.length;
      omitted += sorted.length - shown.length;
      if (shown.length > 0) {
        parts.push(
          `${chain}: ${shown
            .map((e) => residueToken(e.residue, e.atoms ? Array.from(e.atoms) : undefined))
            .join(' ')}`
        );
      }
    });
    return parts.join('; ') + (omitted > 0 ? ` (+${omitted} more)` : '');
  }
}
