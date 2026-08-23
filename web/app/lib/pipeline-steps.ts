export interface PipelineStep {
  id: string;
  label: string;
  dwellMs: number;
}

export const PIPELINE_STEPS: PipelineStep[] = [
  { id: "records", label: "Finding drug records", dwellMs: 400 },
  { id: "documented", label: "Checking documented DrugBank interactions", dwellMs: 700 },
  { id: "overlap", label: "Computing mechanistic overlap", dwellMs: 600 },
  { id: "gapfiller", label: "Running structural gap-filler model", dwellMs: 800 },
  { id: "hgb", label: "Scoring with spectral + HGB classifier", dwellMs: 1500 },
  { id: "gnn", label: "Scoring with GCN link classifier", dwellMs: 1500 },
  { id: "report", label: "Assembling report", dwellMs: 400 },
];
