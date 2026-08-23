export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const DRUGS_TIMEOUT_MS = 15_000;
export const PREDICT_TIMEOUT_MS = 180_000;

export interface PredictedAde {
  ade: string;
  score: number;
}

export interface GroundingChunk {
  drug: string;
  section: string;
  text: string;
}

export type FindingChannel =
  | "documented_outcome"
  | "shared_protein"
  | "risk_axis";

export interface Finding {
  channel: FindingChannel;
  finding: string;
  label?: string;
  outcome?: string;
  proteins?: string[];
  pairs?: string[];
}

export interface Coverage {
  drugs_without_mechanism_data: string[];
  drugs_without_category_data: string[];
  documented_pairs_without_named_outcome?: string[];
  notes: string[];
}

export type InteractionStatus = "documented" | "predicted" | "none_found";

export interface Report {
  interaction_status: InteractionStatus | null;
  pairs_total: number;
  pairs_documented: number;
  pairs_predicted: number;
  interaction_summary: string | null;
  note: string | null;
}

export interface Meta {
  request_id: string;
  timings_ms: Record<string, number>;
  use_real_model: boolean;
}

export interface PredictResponse {
  found: boolean;
  error: string | null;
  error_type: string | null;
  drugs: string[] | null;
  drug_ids: (string | null)[] | null;
  graph_file?: string | null;
  below_training_threshold?: boolean | null;
  predicted_ades: PredictedAde[];
  grounding: GroundingChunk[];
  explanation: string | null;
  findings?: Finding[];
  channels_fired?: string[];
  coverage?: Coverage;
  report: Report | null;
  meta?: Meta;
}

interface FetchOptions {
  timeoutMs?: number;
  signal?: AbortSignal;
}

async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  { timeoutMs = 30_000, signal }: FetchOptions = {}
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);

  const onExternalAbort = () => controller.abort();
  signal?.addEventListener("abort", onExternalAbort);

  try {
    return await fetch(url, {
      ...init,
      signal: controller.signal,
    });
  } catch (err: unknown) {
    if (err instanceof Error && err.name === "AbortError") {
      if (signal?.aborted) {
        throw new Error("Request cancelled.");
      }
      throw new Error(`Request timed out after ${Math.round(timeoutMs / 1000)}s.`);
    }
    throw err;
  } finally {
    window.clearTimeout(timeoutId);
    signal?.removeEventListener("abort", onExternalAbort);
  }
}

export function isSuccessfulAnalysis(response: PredictResponse): boolean {
  if (!response.found) return false;
  if (response.error_type) return false;
  if (!response.report) return false;
  return true;
}

export function analysisErrorMessage(response: PredictResponse): string {
  if (response.error) return response.error;
  if (response.error_type === "drug_not_found") {
    return "One or more drugs were not found in the database.";
  }
  if (response.error_type === "unsupported_drug_count") {
    return "This tool supports 2-3 drugs.";
  }
  return "Unable to complete this analysis.";
}

export async function fetchDrugs(
  options: FetchOptions = {}
): Promise<string[]> {
  const res = await fetchWithTimeout(
    `${API_URL}/drugs`,
    {},
    { timeoutMs: options.timeoutMs ?? DRUGS_TIMEOUT_MS, signal: options.signal }
  );
  if (!res.ok) throw new Error(`/drugs failed: ${res.status}`);
  const data = await res.json();
  return data.drugs ?? [];
}

export async function predict(
  drugs: string[],
  options: FetchOptions = {}
): Promise<PredictResponse> {
  const res = await fetchWithTimeout(
    `${API_URL}/predict`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ drugs }),
    },
    { timeoutMs: options.timeoutMs ?? PREDICT_TIMEOUT_MS, signal: options.signal }
  );
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`/predict failed: ${res.status} ${text}`);
  }
  return res.json();
}
