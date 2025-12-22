use rand::Rng;
use std::collections::VecDeque;

/// Experience tuple
#[derive(Clone)]
pub struct Experience {
    pub state: Vec<f32>,
    pub action: (usize, Vec<f32>),
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
}

/// Prioritized replay buffer
pub struct PrioritizedReplayBuffer {
    capacity: usize,
    buffer: VecDeque<Experience>,
    priorities: Vec<f32>,
    position: usize,
    alpha: f64,
    beta: f64,
}

impl PrioritizedReplayBuffer {
    /// Create new prioritized replay buffer
    pub fn new(capacity: usize, alpha: f64, beta: f64) -> Self {
        Self {
            capacity,
            buffer: VecDeque::with_capacity(capacity),
            priorities: vec![1.0; capacity],
            position: 0,
            alpha,
            beta,
        }
    }

    /// Add experience to buffer
    pub fn add(&mut self, experience: Experience) {
        let max_priority = self.priorities.iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(1.0);

        if self.buffer.len() < self.capacity {
            self.buffer.push_back(experience);
        } else {
            self.buffer[self.position] = experience;
        }

        self.priorities[self.position] = max_priority;
        self.position = (self.position + 1) % self.capacity;
    }

    /// Sample batch from buffer
    pub fn sample(&self, batch_size: usize) -> Option<SampledBatch> {
        if self.buffer.len() < batch_size {
            return None;
        }

        let mut rng = rand::rng();

        // Calculate sampling probabilities
        let priorities: Vec<f32> = self.priorities[..self.buffer.len()]
            .iter()
            .map(|&p| p.powf(self.alpha as f32))
            .collect();

        let sum: f32 = priorities.iter().sum();
        let probs: Vec<f32> = priorities.iter().map(|&p| p / sum).collect();

        // Sample indices
        let mut indices = Vec::with_capacity(batch_size);
        let mut experiences = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let r: f32 = rng.random();
            let mut cumsum = 0.0;
            let mut idx = 0;

            for (i, &prob) in probs.iter().enumerate() {
                cumsum += prob;
                if r <= cumsum {
                    idx = i;
                    break;
                }
            }

            indices.push(idx);
            experiences.push(self.buffer[idx].clone());
        }

        // Calculate importance sampling weights
        let total = self.buffer.len() as f32;
        let weights: Vec<f32> = indices.iter()
            .map(|&idx| {
                let prob = probs[idx];
                (total * prob).powf(-self.beta as f32)
            })
            .collect();

        let max_weight = weights.iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(1.0);

        let normalized_weights: Vec<f32> = weights.iter()
            .map(|&w| w / max_weight)
            .collect();

        Some(SampledBatch {
            experiences,
            indices,
            weights: normalized_weights,
        })
    }

    /// Update priorities based on TD errors
    pub fn update_priorities(&mut self, indices: &[usize], td_errors: &[f32]) {
        for (&idx, &error) in indices.iter().zip(td_errors.iter()) {
            self.priorities[idx] = error.abs() + 1e-6;
        }
    }

    /// Get buffer length
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

/// Sampled batch from replay buffer
pub struct SampledBatch {
    pub experiences: Vec<Experience>,
    pub indices: Vec<usize>,
    pub weights: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replay_buffer() {
        let mut buffer = PrioritizedReplayBuffer::new(100, 0.6, 0.4);

        let exp = Experience {
            state: vec![0.0; 300],
            action: (0, vec![0.0; 6]),
            reward: 1.0,
            next_state: vec![0.0; 300],
            done: false,
        };

        buffer.add(exp);
        assert_eq!(buffer.len(), 1);
    }
}