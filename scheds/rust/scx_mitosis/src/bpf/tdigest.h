/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TDIGEST_H_
#define TDIGEST_H_

#include <stdbool.h>
#include <string.h>
#if !__bpf__
#include <math.h> // for NAN
#include <stddef.h> // for size_t
#include <stdint.h> // for uint64_t
#include <stdlib.h> // for qsort
#endif

struct tdigest_fixedpt {
	uint64_t val;
};

#define FIXEDPT_TOTAL_BITS (64)
#define FIXEDPT_FRAC_BITS (10)
#define FIXEDPT_MAX_U64 (1ull << (FIXEDPT_TOTAL_BITS - FIXEDPT_FRAC_BITS))
#define FIXEDPT_U64(v) \
	((struct tdigest_fixedpt){ .val = v << FIXEDPT_FRAC_BITS })
#define FIXEDPT_AS_U64(f) (f.val >> FIXEDPT_FRAC_BITS)
#if !__bpf__
#define FIXEDPT_DOUBLE(v)          \
	((struct tdigest_fixedpt){ \
		.val = (uint64_t)(v * (1 << FIXEDPT_FRAC_BITS)) })
#define FIXEDPT_AS_DOUBLE(f) (((double)f.val) / (1 << FIXEDPT_FRAC_BITS))
#endif
#define FIXEDPT_HALF \
	((struct tdigest_fixedpt){ .val = 1ull << (FIXEDPT_FRAC_BITS - 1) })
#define FIXEDPT_LT(l, r) (l.val < r.val)
#define FIXEDPT_GT(l, r) (l.val > r.val)
#define FIXEDPT_LTE(l, r) (l.val <= r.val)
#define FIXEDPT_GTE(l, r) (l.val >= r.val)
#define FIXEDPT_ADD(l, r) ((struct tdigest_fixedpt){ .val = l.val + r.val })
#define FIXEDPT_SUB(l, r) ((struct tdigest_fixedpt){ .val = l.val - r.val })
#define FIXEDPT_MUL(l, r)                                    \
	((struct tdigest_fixedpt){ .val = (l.val * r.val) >> \
					  FIXEDPT_FRAC_BITS })
#define FIXEDPT_IMUL(l, r) ((struct tdigest_fixedpt){ .val = (l.val * r) })
#define FIXEDPT_IDIV(l, r) ((struct tdigest_fixedpt){ .val = (l.val / r) })

#define MAX_CENTROIDS (100)

struct t_digest_centroid {
	struct tdigest_fixedpt mean;
	uint64_t weight;
};

struct t_digest {
	struct t_digest_centroid centroids[MAX_CENTROIDS];
	uint64_t numCentroids;
	uint64_t totalWeight;
};

static void t_digest_init(struct t_digest *t_digest)
{
#if __bpf__
	bpf_probe_read_kernel(t_digest, sizeof(*t_digest), NULL);
#else
	memset(t_digest, 0, sizeof(*t_digest));
#endif
}

/*
 * The scaling function used here is...
 *   k(q, d) = (IF q >= 0.5, d - d * sqrt(2 - 2q) / 2, d * sqrt(2q) / 2)
 *
 *   k(0, d) = 0
 *   k(1, d) = d
 *
 *   where d is the compression value
 */
static struct tdigest_fixedpt k_to_q(uint64_t k)
{
	// k_div_d = k / compression
	struct tdigest_fixedpt k_div_d =
		FIXEDPT_IDIV(FIXEDPT_U64(k), MAX_CENTROIDS);
	// if (k_div_d < 0.5)
	if (FIXEDPT_GTE(k_div_d, FIXEDPT_HALF)) {
		// base = 1 - k_div_d
		struct tdigest_fixedpt base =
			FIXEDPT_SUB(FIXEDPT_U64(1), k_div_d);
		// return 1 - base * base * 2
		return FIXEDPT_SUB(FIXEDPT_U64(1),
				   FIXEDPT_IMUL(FIXEDPT_MUL(base, base), 2));
	} else {
		// return k_div_d * k_div_d * 2
		return FIXEDPT_IMUL(FIXEDPT_MUL(k_div_d, k_div_d), 2);
	}
}

static void t_digest_add(struct t_digest *t_digest, uint64_t value)
{
	if (t_digest->numCentroids >= MAX_CENTROIDS) {
		// We look to merge only two adjacent centroids so that the new value can be
		// inserted
		uint64_t weightSoFar = t_digest->centroids[0].weight;
		bool merged = false;
		for (int i = 1; i < MAX_CENTROIDS; ++i) {
			struct t_digest_centroid *prev =
				&t_digest->centroids[i - 1];
			struct t_digest_centroid *cur = &t_digest->centroids[i];
			if (!merged) {
				weightSoFar += cur->weight;
				// Add one to totalWeight here to account for the additional value
				struct tdigest_fixedpt q_limit_times_weight =
					FIXEDPT_IMUL(k_to_q(i),
						     t_digest->totalWeight + 1);
				if (FIXEDPT_LTE(FIXEDPT_U64(weightSoFar),
						q_limit_times_weight)) {
					prev->weight += cur->weight;
					struct tdigest_fixedpt delta =
						FIXEDPT_SUB(cur->mean,
							    prev->mean);
					struct tdigest_fixedpt weighted_delta =
						FIXEDPT_IDIV(
							FIXEDPT_IMUL(
								delta,
								cur->weight),
							prev->weight);
					prev->mean = FIXEDPT_ADD(
						prev->mean, weighted_delta);
					merged = true;
				}
			} else {
				// We already found and merged centroids, so just move the rest down
				prev->mean = cur->mean;
				prev->weight = cur->weight;
			}
		}
		if (!merged) {
			// We must have hit some logic bug
			return;
		}
		t_digest->numCentroids = MAX_CENTROIDS - 1;
	}

	// The state now should be that we have numCentroids < MAX_CENTROIDS, so
	// just insert the new centroid into sorted order.
	struct t_digest_centroid insert = (struct t_digest_centroid){
		.mean = FIXEDPT_U64(value),
		.weight = 1,
	};
	// The additional MAX_CENTROIDS check in this loop conditional shouldn't
	// be necessary, but the verifier needs it
	for (int i = 0; i < t_digest->numCentroids && i < MAX_CENTROIDS; ++i) {
		struct t_digest_centroid *cur = &t_digest->centroids[i];
		if (FIXEDPT_LTE(insert.mean, cur->mean)) {
			struct tdigest_fixedpt temp_mean;
			temp_mean.val = cur->mean.val;
			uint64_t temp_weight = cur->weight;
			cur->mean = insert.mean;
			cur->weight = insert.weight;
			insert.mean.val = temp_mean.val;
			insert.weight = temp_weight;
		}
	}
	// Should not be possible, but the verifier needs this check
	if (t_digest->numCentroids >= MAX_CENTROIDS) {
		return;
	}
	struct t_digest_centroid *end =
		&t_digest->centroids[t_digest->numCentroids];
	end->mean.val = insert.mean.val;
	end->weight = insert.weight;
	t_digest->numCentroids++;
	t_digest->totalWeight++;
}

#if !__bpf__
static int tdigest_compare_centroids(const void *l, const void *r)
{
	const struct t_digest_centroid *lc =
		(const struct t_digest_centroid *)l;
	const struct t_digest_centroid *rc =
		(const struct t_digest_centroid *)r;
	if (FIXEDPT_LT(lc->mean, rc->mean))
		return -1;
	else if (FIXEDPT_GT(lc->mean, rc->mean))
		return 1;
	return 0;
}

// Merge t_digest r into t_digest l
static void tdigest_merge_digests(struct t_digest *l, struct t_digest *r)
{
	if (!l->numCentroids && !r->numCentroids)
		return;

	// First combine the two sets of centroids and sort them
	struct t_digest_centroid centroids[MAX_CENTROIDS * 2];
	memcpy(centroids, l->centroids,
	       sizeof(struct t_digest_centroid) * l->numCentroids);
	memcpy(&centroids[l->numCentroids], r->centroids,
	       sizeof(struct t_digest_centroid) * r->numCentroids);
	uint64_t numCentroids = l->numCentroids + r->numCentroids;
	qsort((void *)centroids, numCentroids, sizeof(struct t_digest_centroid),
	      &tdigest_compare_centroids);

	uint64_t totalWeight = l->totalWeight + r->totalWeight;
	int cur = 0;
	uint64_t weightSoFar = centroids[0].weight;
	uint64_t k_limit = 1;
	double q_limit_times_weight =
		FIXEDPT_AS_DOUBLE(k_to_q(k_limit++)) * totalWeight;
	for (int i = 1; i < numCentroids; ++i) {
		weightSoFar += centroids[i].weight;
		if (weightSoFar <= q_limit_times_weight) {
			centroids[cur].weight += centroids[i].weight;
			double delta = FIXEDPT_AS_DOUBLE(centroids[i].mean) -
				       FIXEDPT_AS_DOUBLE(centroids[cur].mean);
			double weightedDelta = delta * centroids[i].weight /
					       centroids[cur].weight;
			centroids[cur].mean = FIXEDPT_DOUBLE(
				FIXEDPT_AS_DOUBLE(centroids[cur].mean) +
				weightedDelta);
		} else {
			// cur is full, advance to the next centroid
			cur++;
			centroids[cur] = centroids[i];
			q_limit_times_weight =
				FIXEDPT_AS_DOUBLE(k_to_q(k_limit++)) *
				totalWeight;
		}
		if (cur != i) {
			centroids[i] = (struct t_digest_centroid){
				.mean = FIXEDPT_U64(0),
				.weight = 0,
			};
		}
	}
	l->numCentroids = cur + 1;
	l->totalWeight = totalWeight;
	memcpy(&l->centroids, centroids,
	       sizeof(struct t_digest_centroid) * l->numCentroids);
}

// Get an estimate for the value at the quantile q (e.g. q = 0.99 means to get
// the 99th percentile)
static double t_digest_value_at(struct t_digest *t_digest, double q)
{
	if (q > 1.0 || q < 0.0 || !t_digest->numCentroids) {
		return NAN;
	}
	double goal = q * t_digest->totalWeight;
	double weightSoFar = 0;
	size_t i = 0;
	const struct t_digest_centroid *c;
	// Identify the first centroid past our quantile goal
	for (i = 0; i < t_digest->numCentroids; ++i) {
		c = &t_digest->centroids[i];
		if (weightSoFar + c->weight > goal)
			break;
		weightSoFar += c->weight;
	}
	double deltaWeightSoFar = goal - weightSoFar - c->weight / 2.0;
	// are we closer to the right centroid?
	bool right = deltaWeightSoFar > 0.0;
	if ((right && ((i + 1) == t_digest->numCentroids)) || (!right && !i)) {
		// If we are before the first centroid or past the last centroid, just
		// use it
		return FIXEDPT_AS_DOUBLE(c->mean);
	}

	// Otherwise, we interpolate
	const struct t_digest_centroid *cl;
	const struct t_digest_centroid *cr;
	if (right) {
		cl = c;
		cr = &t_digest->centroids[i + 1];
		weightSoFar += cl->weight / 2.0;
	} else {
		cl = &t_digest->centroids[i - 1];
		cr = c;
		weightSoFar -= cl->weight / 2.0;
	}
	double x = goal - weightSoFar;
	double m = (FIXEDPT_AS_DOUBLE(cr->mean) - FIXEDPT_AS_DOUBLE(cl->mean)) /
		   (cl->weight / 2.0 + cr->weight / 2.0);
	return m * x + FIXEDPT_AS_DOUBLE(cl->mean);
}
#endif

#undef FIXEDPT_TOTAL_BITS
#undef FIXEDPT_FRAC_BITS
#undef FIXEDPT_MAX_U64
#undef FIXEDPT_U64
#undef FIXEDPT_AS_U64
#if !__bpf__
#undef FIXEDPT_DOUBLE
#undef FIXEDPT_AS_DOUBLE
#endif
#undef FIXEDPT_HALF
#undef FIXEDPT_LT
#undef FIXEDPT_GT
#undef FIXEDPT_LTE
#undef FIXEDPT_GTE
#undef FIXEDPT_ADD
#undef FIXEDPT_SUB
#undef FIXEDPT_MUL
#undef FIXEDPT_IMUL
#undef FIXEDPT_IDIV

#undef MAX_CENTROIDS

#endif // TDIGEST_H_
